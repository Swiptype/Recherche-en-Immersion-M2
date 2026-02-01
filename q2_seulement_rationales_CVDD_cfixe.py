import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
import csv

# --- Configuration ---
DEVICE = torch.device('cuda')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LATENT_DIM = 64
MAX_LEN = 128
UNFREEZE_AFTER_EPOCH = 2
RATIONALE_PERCENTAGE = 0.20
NUM_RUNS = 5 
SUPER_CATEGORIES = ['alt', 'comp', 'misc', 'rec', 'sci', 'soc', 'talk']

FILE_SUMMARY = "results_root_exp2_summary.csv"
FILE_DETAILS = "results_root_exp2_details.csv"

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
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'raw_text': text}

def train_and_evaluate(super_cat, run_id, all_data, all_targets, target_names):
    print(f"   Run {run_id+1}/{NUM_RUNS}...")
    
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
    test_normal = normal_data[split_idx:]
    
    n_anomalies = int(len(test_normal) / 9)
    if n_anomalies > len(anomaly_data): n_anomalies = len(anomaly_data)
    test_anomalies = random.sample(anomaly_data, n_anomalies)
    
    test_texts = test_normal + test_anomalies
    test_labels = [0] * len(test_normal) + [1] * len(test_anomalies)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader = DataLoader(TextDataset(train_normal, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TextDataset(test_texts, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    
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
    scores_baseline = []
    scores_only = []
    details_log = []
    
    def forward_score(inputs_embeds, attention_mask):
        z, c = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return torch.sum((z - c) ** 2, dim=1)
    
    ig = IntegratedGradients(forward_score)
    
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            z, c = model(input_ids=input_ids, attention_mask=mask)
            scores_baseline.append(torch.sum((z - c) ** 2, dim=1).item())
            
        embeddings_list = model.bert.embeddings.word_embeddings(input_ids)
        attributions = ig.attribute(inputs=embeddings_list, additional_forward_args=(mask), n_steps=100)
        token_importance = attributions.sum(dim=2)
        
        k = int(MAX_LEN * RATIONALE_PERCENTAGE)
        top_val, top_idx = torch.topk(token_importance, k, dim=1)
        top_indices = top_idx[0].tolist()
        
        modified_input_ids = torch.zeros_like(input_ids)
        kept_tokens = []
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        for idx in top_indices:
            modified_input_ids[0, idx] = input_ids[0, idx]
            if idx < len(all_tokens):
                kept_tokens.append(all_tokens[idx])
            
        with torch.no_grad():
            z, c = model(input_ids=modified_input_ids, attention_mask=mask)
            scores_only.append(torch.sum((z - c) ** 2, dim=1).item())
            
        is_anomaly = test_labels[i] == 1
        if i < 5 or (is_anomaly and i < (len(test_normal) + 5)):
            details_log.append([super_cat, run_id, i, "Anomaly" if is_anomaly else "Normal", scores_baseline[-1], scores_only[-1], kept_tokens])

    try:
        auc_base = roc_auc_score(test_labels, scores_baseline)
        auc_only = roc_auc_score(test_labels, scores_only)
    except ValueError:
        auc_base = 0.5
        auc_only = 0.5
    
    return auc_base, auc_only, details_log

def main():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_data = newsgroups.data
    all_targets = newsgroups.target
    target_names = newsgroups.target_names
    
    with open(FILE_SUMMARY, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SuperCategory", "AUC_Base_Mean", "AUC_Base_Std", "AUC_OnlyRat_Mean", "AUC_OnlyRat_Std"])
        
    with open(FILE_DETAILS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["SuperCategory", "Run_ID", "Sample_ID", "Label", "Score_Original", "Score_OnlyRat", "Kept_Rationales"])

    for super_cat in SUPER_CATEGORIES:
        print(f"Processing Super-Category: {super_cat}")
        base_aucs, only_aucs = [], []
        
        for run in range(NUM_RUNS):
            ab, ao, logs = train_and_evaluate(super_cat, run, all_data, all_targets, target_names)
            base_aucs.append(ab)
            only_aucs.append(ao)
            
            with open(FILE_DETAILS, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(logs)
        
        mean_base = np.mean(base_aucs)
        std_base = np.std(base_aucs)
        mean_only = np.mean(only_aucs)
        std_only = np.std(only_aucs)
        
        with open(FILE_SUMMARY, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([super_cat, mean_base, std_base, mean_only, std_only])
            
        print(f"  > Done {super_cat}. Base: {mean_base:.3f}, Only: {mean_only:.3f}")

if __name__ == "__main__":
    main()
