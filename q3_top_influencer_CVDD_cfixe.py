import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.datasets import fetch_20newsgroups
import random
import pandas as pd
from collections import defaultdict
from captum.attr import IntegratedGradients
import csv

# --- Config ---
DEVICE = torch.device('cuda')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LATENT_DIM = 64
MAX_LEN = 128
UNFREEZE_AFTER_EPOCH = 2
NUM_RUNS = 5
SUPER_CATEGORIES = ['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']

FILE_TOKENS = "results_root_exp3_tokens.csv"

# --- Modèle CVDD ---
class CVDD(nn.Module):
    def __init__(self):
        super(CVDD, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Réseau principal pour la projection
        self.projection = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(256, LATENT_DIM)
        )
        self.c = nn.Parameter(torch.zeros(1, LATENT_DIM), requires_grad=False)
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
            cls_output = outputs.last_hidden_state[:, 0, :]
            z = self.projection(cls_output)
            return z, self.c.expand(z.size(0), -1)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        text = " ".join(str(self.texts[item]).split()) 
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

def run_analysis(super_cat, all_data, all_targets, target_names, tokenizer):
    # --- LOGIQUE DE GROUPEMENT PAR RACINE ---
    normal_data = []
    anomaly_data = []
    
    for text, target_idx in zip(all_data, all_targets):
        cat_name = target_names[target_idx]
        if cat_name.startswith(super_cat):
            normal_data.append(text)
        else:
            anomaly_data.append(text)
    
    split_idx = int(0.8 * len(normal_data))
    train_normal = normal_data[:split_idx]
    
    # Echantillon pour l'analyse (100 de chaque pour être rapide mais représentatif)
    analyze_normal = normal_data[split_idx:split_idx+100]
    if len(analyze_normal) < 10: analyze_normal = normal_data[split_idx:]
    
    analyze_anomalies = random.sample(anomaly_data, min(100, len(anomaly_data)))
    
    train_loader = DataLoader(TextDataset(train_normal, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    loader_normal = DataLoader(TextDataset(analyze_normal, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    loader_anomaly = DataLoader(TextDataset(analyze_anomalies, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    
    model = CVDD().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Entraînement CVDD
    for epoch in range(NUM_EPOCHS):
        model.train()
        if epoch >= UNFREEZE_AFTER_EPOCH:
            for param in model.bert.parameters(): 
                param.requires_grad = True
        else:
            for param in model.bert.parameters(): 
                param.requires_grad = False
        
        for batch in train_loader:
            optimizer.zero_grad()
            z, c = model(input_ids=batch['input_ids'].to(DEVICE), 
                        attention_mask=batch['attention_mask'].to(DEVICE))
            # Loss CVDD: distance entre z et son centre contextuel c
            loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()

    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            z, _ = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            all_vecs.append(z)
            if i > 10: break
    model.c.data = torch.mean(torch.cat(all_vecs), dim=0, keepdim=True)
    
    def forward_score(inputs_embeds, attention_mask):
        z, c = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return torch.sum((z - c) ** 2, dim=1)
    
    ig = IntegratedGradients(forward_score)
    
    local_impact_anomalies = defaultdict(float)
    
    # On analyse surtout les Anomalies pour voir ce qui "saute aux yeux" du modèle
    # (Quels mots dans un texte non-comp le rendent très éloigné du centre comp ?)
    for batch in loader_anomaly:
        input_ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
        
        attributions = ig.attribute(inputs=embeddings, additional_forward_args=(mask), n_steps=100)
        token_imp = attributions.sum(dim=2).squeeze(0)
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        for idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']: continue
            local_impact_anomalies[token] += token_imp[idx].item()
                
    return local_impact_anomalies

def main():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_data = newsgroups.data
    all_targets = newsgroups.target
    target_names = newsgroups.target_names
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    with open(FILE_TOKENS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["SuperCategory", "TokenType", "Token", "AccumulatedScore"])

    for super_cat in SUPER_CATEGORIES:
        print(f"Processing Super-Category: {super_cat}")
        
        global_anom = defaultdict(float)
        
        for run in range(NUM_RUNS):
            print(f"   Run {run+1}/{NUM_RUNS}")
            anom = run_analysis(super_cat, all_data, all_targets, target_names, tokenizer)
            for k, v in anom.items(): global_anom[k] += v
            
        # Top 50 mots qui "causent" l'anomalie
        # (C'est à dire: quand le modèle s'attend à du Sport (Rec), quels mots le choquent le plus ?)
        sorted_anom = sorted(global_anom.items(), key=lambda x: x[1], reverse=True)[:50]
        
        with open(FILE_TOKENS, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for token, score in sorted_anom:
                writer.writerow([super_cat, "Anomaly_Driver", token, score/NUM_RUNS])

if __name__ == "__main__":
    main()
