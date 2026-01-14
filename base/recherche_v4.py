import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score, average_precision_score
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import os

# --- Configuration Globale ---
MAX_LEN = 128           
BATCH_SIZE = 16         
EPOCHS = 8              
LEARNING_RATE = 2e-5    
EMBED_DIM = 768         
HIDDEN_DIM = 256        
NU = 0.1                
LAMBDA_REG = 1.0       
UNFREEZE_AFTER_EPOCH = 2
NUM_RUNS = 5            # Nombre d'itérations par catégorie pour l'écart-type

# --- 1. Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        # Filtrage des textes trop courts
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

# --- 2. Modèle CVDD (Deep SVDD avec Attention) ---
class TextAnomalyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Gèle initial
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Couche d'attention pour agréger les sorties
        self.attention_query = nn.Linear(EMBED_DIM, HIDDEN_DIM)
        
        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )

        self.c = nn.Parameter(torch.randn(1, HIDDEN_DIM), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # Support pour inputs_embeds nécessaire pour Integrated Gradients
        outputs = self.distilbert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds
        )
        last_hidden = outputs.last_hidden_state # [B, T, D]
        
        # Mean Pooling pondéré (simplifié pour la stabilité du gradient)
        vec = torch.mean(last_hidden, dim=1) 
        
        projected = self.projection(vec)
        # Score = Distance euclidienne carrée au centre c
        score = torch.sum((projected - self.c) ** 2, dim=1)
        return score

# --- 3. Explainer avec Integrated Gradients ---
class IGExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        # Wrapper pour Captum
        def forward_func(inputs_embeds, attention_mask):
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.ig = IntegratedGradients(forward_func)

    def explain(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)

        # Baseline : embeddings nuls (ou token neutre)
        input_embeds = self.model.distilbert.embeddings(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        # Calcul des attributions
        attrs = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            additional_forward_args=(mask,),
            n_steps=50
        )

        # Somme sur la dimension d'embedding pour avoir un score par mot
        word_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        with torch.no_grad():
            score = self.model(input_ids=input_ids, attention_mask=mask).item()

        return tokens, word_attrs, score

# --- 4. Fonction d'entraînement d'un Run ---
def train_one_run(normal_cat_name, all_categories, device):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Chargement One-vs-Rest
    train_txt = fetch_20newsgroups(subset='train', categories=[normal_cat_name], remove=('headers', 'footers', 'quotes')).data
    test_all = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # Labels: 0 pour Normal, 1 pour Anomaly
    y_true = [0 if all_categories[target] == normal_cat_name else 1 for target in test_all.target]
    
    train_loader = DataLoader(TextDataset(train_txt, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    model = TextAnomalyModel().to(device)

    # Initialisation de C
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for batch in train_loader:
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            # Récupération de la projection manuelle pour initialiser C
            out = model.distilbert(inp, attention_mask=mask).last_hidden_state
            proj = model.projection(torch.mean(out, dim=1))
            all_vecs.append(proj)
            if len(all_vecs) > 10: break
    model.c.data = torch.mean(torch.cat(all_vecs), dim=0, keepdim=True)

    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Boucle d'entraînement
    model.train()
    for epoch in range(EPOCHS):
        if epoch == UNFREEZE_AFTER_EPOCH:
            for param in model.distilbert.parameters(): param.requires_grad = True
            
        for batch in train_loader:
            optimizer.zero_grad()
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            scores = model(inp, mask)
            loss = torch.mean(scores) # SVDD objective
            loss.backward()
            optimizer.step()

    # Évaluation
    model.eval()
    y_scores = []
    with torch.no_grad():
        for text in test_all.data:
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
            s = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).item()
            y_scores.append(s)
            
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return auc, ap

# --- 5. Boucle Principale de Benchmark ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_categories = fetch_20newsgroups(subset='all').target_names
    results_list = []

    print(f"Lancement du benchmark sur 20 classes avec {NUM_RUNS} runs chacune...")

    for cat in all_categories:
        aucs, aps = [], []
        print(f"\nClasse Normale Actuelle: {cat}")
        
        for r in range(NUM_RUNS):
            auc, ap = train_one_run(cat, all_categories, device)
            aucs.append(auc)
            aps.append(ap)
            print(f"  Run {r+1}/{NUM_RUNS} - AUC: {auc:.4f}")
        
        results_list.append({
            'Category': cat,
            'AUC_mean': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'AP_mean': np.mean(aps),
            'AP_std': np.std(aps)
        })

    # Sauvegarde finale
    df = pd.DataFrame(results_list)
    df.to_csv("cvdd_20news_results.csv", index=False)
    print("\n" + "="*30)
    print("Tableau Récapitulatif :")
    print(df[['Category', 'AUC_mean', 'AUC_std']])

if __name__ == "__main__":
    main()