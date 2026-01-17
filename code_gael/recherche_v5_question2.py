import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from captum.attr import IntegratedGradients
from colorama import Fore, Back, Style, init
import numpy as np
import pandas as pd
import os
import re

# Initialisation
init(autoreset=True)

# --- Configuration ---
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
EMBED_DIM = 768
HIDDEN_DIM = 256
NU = 0.1
UNFREEZE_AFTER_EPOCH = 2
NUM_RUNS = 5
OUTPUT_FILE = "rationales_question2.txt"

# --- 1. Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [t for t in texts if isinstance(t, str) and len(t) > 50]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- 2. Modèle ---
class TextAnomalyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.distilbert.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        self.c = nn.Parameter(torch.randn(1, HIDDEN_DIM), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        last_hidden = outputs.last_hidden_state 
        vec = last_hidden[:, 0, :] # Token [CLS]
        projected = self.projection(vec)
        score = torch.sum((projected - self.c) ** 2, dim=1)
        return score

# --- 3. Explainer ---
class IGExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        def forward_func(inputs_embeds, attention_mask):
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.ig = IntegratedGradients(forward_func)

    def explain_to_file(self, text, threshold, file_path=OUTPUT_FILE):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        input_embeds = self.model.distilbert.embeddings(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        attrs = self.ig.attribute(inputs=input_embeds, baselines=baseline_embeds, additional_forward_args=(mask,), n_steps=32)
        word_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        # Normalisation
        attrs_norm = np.clip(word_attrs, 0, None)
        if attrs_norm.max() > 0: attrs_norm /= attrs_norm.max()

        # Listes de filtrage
        CUSTOM_STOP = set(['the', 'a', 'an', 'and', 'or', 'if', 'is', 'are', 'was', 'were', 
                           'of', 'to', 'in', 'for', 'on', 'at', 'by', 'from', 'with', 
                           'that', 'this', 'it', 'as', 'be', 'have', 'has', 'but', 'not'])
        SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']

        extracted_rationales = []
        for token, weight in zip(tokens, attrs_norm):
            if token in SPECIAL_TOKENS: continue
            clean_t = token.replace("##", "")
            check_t = clean_t.lower()
            
            is_stop = (check_t in ENGLISH_STOP_WORDS) or (check_t in CUSTOM_STOP) or len(check_t) < 2
            
            if not is_stop and weight > 0.1: 
                extracted_rationales.append((clean_t, weight))

        # Écriture dans le fichier
        extracted_rationales.sort(key=lambda x: x[1], reverse=True)
        with open(file_path, "a", encoding="utf-8") as f:
            for word, importance in extracted_rationales[:10]: # Top 10 par texte
                f.write(f"{word}\n")

# --- 4. Fonctions de Suffisance (Garder UNIQUEMENT les rationales) ---

def get_rationale_words(file_path):
    """Lit le fichier rationales.txt et crée un set de mots À GARDER (Whitelist)"""
    if not os.path.exists(file_path):
        return set()
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    words = set()
    for line in content.splitlines():
        clean = line.strip().lower()
        if clean and not clean.startswith("---") and not clean.startswith("Rat"):
            words.add(clean)
    
    print(f"   -> {len(words)} rationales chargés. Ce sont les SEULS mots qui seront gardés.")
    return words

def evaluate_with_sufficiency(model, tokenizer, test_data, y_true, device, file_path):
    """
    Supprime TOUS les mots SAUF ceux listés dans le fichier (Test de Suffisance).
    """
    whitelist = get_rationale_words(file_path)
    if not whitelist:
        return 0.0

    model.eval()
    y_scores_only_rationales = []
    
    print("   -> Recalcul des scores en gardant UNIQUEMENT les rationales...")
    
    for text in test_data:
        # Nettoyage inversé
        words = text.split()
        # On garde le mot SEULEMENT s'il est dans la whitelist (les rationales)
        kept_words = [w for w in words if w.lower().strip(",.") in whitelist]
        
        clean_text = " ".join(kept_words)
        
        # Si la phrase devient vide (aucun rationale trouvé), on met un token [PAD]
        # Dans ce cas, le modèle devrait donner un score très bas (Normal), car il n'y a plus d'anomalie.
        if not clean_text.strip(): 
            clean_text = "[PAD]"

        # Passage modèle
        inputs = tokenizer(clean_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            score = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).item()
            y_scores_only_rationales.append(score)

    return roc_auc_score(y_true, y_scores_only_rationales)

# --- 5. Boucle Principale ---
def train_and_eval(normal_cat_name, all_categories, device, run_idx):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_txt = fetch_20newsgroups(subset='train', categories=[normal_cat_name], remove=('headers', 'footers', 'quotes')).data
    test_all = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    y_true = [0 if all_categories[target] == normal_cat_name else 1 for target in test_all.target]
    
    train_loader = DataLoader(TextDataset(train_txt, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    model = TextAnomalyModel().to(device)

    # Init Center
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for batch in train_loader:
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            out = model.distilbert(inp, attention_mask=mask).last_hidden_state
            vec = out[:, 0, :]
            proj = model.projection(vec)
            all_vecs.append(proj)
            if len(all_vecs) > 10: break
    model.c.data = torch.mean(torch.cat(all_vecs), dim=0, keepdim=True)

    # Train
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        if epoch == UNFREEZE_AFTER_EPOCH:
            for p in model.distilbert.parameters(): p.requires_grad = True
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            scores = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()

    # Eval 1: Original
    model.eval()
    y_scores = []
    with torch.no_grad():
        for text in test_all.data:
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
            y_scores.append(model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).item())
    
    auc_original = roc_auc_score(y_true, y_scores)

    # --- PHASE SUFFISANCE (ONLY RATIONALES) ---
    auc_sufficiency = 0.0
    
    if run_idx == NUM_RUNS - 1:
        if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
        
        # Identifier les anomalies pour extraire les rationales
        indices_anomalies = []
        for i, (label, score) in enumerate(zip(y_true, y_scores)):
            if label == 1: 
                indices_anomalies.append((i, score))
        
        indices_anomalies.sort(key=lambda x: x[1], reverse=True)
        top_anomalies = indices_anomalies[:5] # On extrait les rationales des 5 pires anomalies

        explainer = IGExplainer(model, tokenizer)
        threshold = np.quantile(y_scores, 1 - NU)

        print(f"   -> Extraction des rationales vers {OUTPUT_FILE}...")
        for idx, _ in top_anomalies:
            explainer.explain_to_file(test_all.data[idx], threshold)

        # 3. Recalculer l'AUC en ne gardant QUE ces mots
        auc_sufficiency = evaluate_with_sufficiency(model, tokenizer, test_all.data, y_true, device, OUTPUT_FILE)
    else:
        auc_sufficiency = auc_original 

    return auc_original, auc_sufficiency

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_cats = fetch_20newsgroups(subset='all').target_names 
    
    final_stats = []
    categories_to_test = all_cats 

    for cat in categories_to_test:
        aucs_orig = []
        aucs_suff = []
        print(f"\nTarget Normal: {cat}")
        
        for r in range(NUM_RUNS):
            auc, auc_suf = train_and_eval(cat, all_cats, device, r)
            aucs_orig.append(auc)
            if r == NUM_RUNS - 1: 
                aucs_suff.append(auc_suf)
            
            print(f" Run {r+1}: AUC Originale {auc:.4f}")
            if r == NUM_RUNS - 1:
                print(f" -> AUC Seulement Rationales (Run Final): {auc_suf:.4f}")

        # Moyenne
        final_stats.append({
            'Category': cat, 
            'AUC_Orig_Mean': np.mean(aucs_orig), 
            'AUC_OnlyRat_Final': aucs_suff[0] if aucs_suff else 0,
            # Ici on regarde la difference. Si c'est proche de 0, c'est que les rationales suffisent.
            'Difference': np.mean(aucs_orig) - (aucs_suff[0] if aucs_suff else 0)
        })

    df = pd.DataFrame(final_stats)
    df.to_csv("benchmark_sufficiency_results.csv", index=False)
    print("\n--- RÉSULTATS COMPARATIFS (Orig vs Only Rationales) ---\n")
    print(df[['Category', 'AUC_Orig_Mean', 'AUC_OnlyRat_Final', 'Difference']])

if __name__ == "__main__":
    main()
