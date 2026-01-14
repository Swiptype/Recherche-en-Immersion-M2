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

# Initialisation pour les couleurs en console
init(autoreset=True)

# --- Configuration Globale ---
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
EMBED_DIM = 768
HIDDEN_DIM = 256
NU = 0.1
UNFREEZE_AFTER_EPOCH = 2
NUM_RUNS = 5
OUTPUT_FILE = "rationales.txt" # Nom du fichier de sortie

# --- 1. Dataset  ---
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

# --- 2. Modèle CVDD (Deep SVDD) ---
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
        # Utilisation du token [CLS] (index 0) pour représenter la phrase entière
        vec = last_hidden[:, 0, :]
        projected = self.projection(vec)
        score = torch.sum((projected - self.c) ** 2, dim=1)
        return score

# --- 3. Explainer & Visualizer  ---
class IGExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        def forward_func(inputs_embeds, attention_mask):
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.ig = IntegratedGradients(forward_func)

    def explain_and_visualize(self, text, threshold, file_path=OUTPUT_FILE):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)

        input_embeds = self.model.distilbert.embeddings(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        # Calcul des gradients (Integrated Gradients)
        attrs = self.ig.attribute(inputs=input_embeds, baselines=baseline_embeds, additional_forward_args=(mask,), n_steps=32)
        word_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        with torch.no_grad():
            score = self.model(input_ids=input_ids, attention_mask=mask).item()

        # Normalisation pour la visualisation
        attrs_norm = np.clip(word_attrs, 0, None)
        if attrs_norm.max() > 0: attrs_norm /= attrs_norm.max()

        status = '❌ ANOMALIE' if score > threshold else '✅ NORMAL'
        print(f"\nScore: {score:.4f} (Seuil: {threshold:.4f}) | {status}")
        
        # Liste élargie de Stop Words + Tokens spéciaux à ignorer pour l'extraction
        CUSTOM_STOP = set(['the', 'a', 'an', 'and', 'or', 'if', 'is', 'are', 'was', 'were', 
                           'of', 'to', 'in', 'for', 'on', 'at', 'by', 'from', 'with', 
                           'that', 'this', 'it', 'as', 'be', 'have', 'has', 'but', 'not',
                           'can', 'will', 'do', 'my', 'your', 'we', 'they', 'he', 'she'])
        SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']

        # --- 1. Visualisation Console ---
        colored_text = []
        extracted_rationales = [] # Liste de tuples (mot, score)

        for token, weight in zip(tokens, attrs_norm):
            if token in SPECIAL_TOKENS: continue
            
            clean_t = token.replace("##", "")
            check_t = clean_t.lower()

            # On stocke tout ce qui n'est pas un stop word pour le tri final
            is_stop_word = (check_t in ENGLISH_STOP_WORDS) or (check_t in CUSTOM_STOP) or len(check_t) < 2
            
            if not is_stop_word:
                # On ajoute le mot et son poids brut (word_attrs) ou normalisé (weight) à la liste
                # Ici on prend le poids normalisé pour le classement
                extracted_rationales.append((clean_t, weight))

            # Logique de couleur (Console uniquement)
            if is_stop_word:
                color = Style.DIM
            elif weight > 0.7: color = Back.RED + Fore.WHITE
            elif weight > 0.4: color = Fore.RED + Style.BRIGHT
            elif weight > 0.1: color = Fore.YELLOW
            else: color = Style.DIM
            
            colored_text.append(color + clean_t + Style.RESET_ALL)
        
        print(" ".join(colored_text))

        # --- 2. Écriture dans le fichier rationales.txt ---
        # Tri des mots par importance décroissante (plus grand poids en premier)
        extracted_rationales.sort(key=lambda x: x[1], reverse=True)

        print(f"   -> Extraction des rationales vers {file_path}...")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- ANOMALY DETECTED (Score: {score:.4f}) ---\n")
            f.write("Rationales (Top words contributing to anomaly):\n")
            # On écrit les 10-15 plus importants
            for word, importance in extracted_rationales[:15]:
                if importance > 0: # On n'écrit que ce qui a un impact positif
                    f.write(f"{word}\n")
            f.write("-" * 30 + "\n")

# --- 4. Logiciel de Benchmark ---
def train_and_eval(normal_cat_name, all_categories, device, run_idx):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_txt = fetch_20newsgroups(subset='train', categories=[normal_cat_name], remove=('headers', 'footers', 'quotes')).data
    test_all = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    y_true = [0 if all_categories[target] == normal_cat_name else 1 for target in test_all.target]
    
    train_loader = DataLoader(TextDataset(train_txt, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    model = TextAnomalyModel().to(device)

    # Init Center C 
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for batch in train_loader:
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            out = model.distilbert(inp, attention_mask=mask).last_hidden_state
            # On doit prendre le token [CLS] (index 0) et pas la moyenne, 
            # pour être cohérent avec la fonction forward()
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

    # Eval
    model.eval()
    y_scores = []
    with torch.no_grad():
        for text in test_all.data:
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
            y_scores.append(model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).item())
    
    auc = roc_auc_score(y_true, y_scores)
    
    # Visualisation sur un échantillon d'anomalie au dernier run
    if run_idx == NUM_RUNS - 1:
        threshold = np.quantile(y_scores, 1 - NU)
        explainer = IGExplainer(model, tokenizer)
        # On cherche une vraie anomalie détectée (Score > Seuil ET Label = 1)
        idx_anom = -1
        for i, (true_label, score) in enumerate(zip(y_true, y_scores)):
             if true_label == 1 and score > threshold:
                 idx_anom = i
                 break
        
        if idx_anom != -1:
            explainer.explain_and_visualize(test_all.data[idx_anom], threshold)

    return auc

def main():
    # Reset du fichier rationales au début
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # On limite à quelques catégories pour le test si besoin, sinon 'all'
    all_cats = fetch_20newsgroups(subset='all').target_names 
    
    final_stats = []

    # Pour tester rapidement, tu peux réduire la boucle ici (ex: all_cats[:2])
    for cat in all_cats:
        aucs = []
        print(f"\nTarget Normal: {cat}")
        for r in range(NUM_RUNS):
            auc = train_and_eval(cat, all_cats, device, r)
            aucs.append(auc)
            print(f" Run {r+1}: AUC {auc:.4f}")
        
        final_stats.append({'Category': cat, 'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs)})

    df = pd.DataFrame(final_stats)
    df.to_csv("benchmark_results.csv", index=False)
    print("\n--- RÉSULTATS FINAUX ---\n", df)

if __name__ == "__main__":
    main()
