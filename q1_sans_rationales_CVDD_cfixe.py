import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
import numpy as np
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

FILE_SUMMARY = "results_root_exp1_summary.csv"
FILE_DETAILS = "results_root_exp1_details.csv"

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
        if inputs_embeds is not None:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Projection dans l'espace latent
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
        # Si la catégorie commence par le prefixe (ex: 'sci.space' commence par 'sci') -> NORMAL
        if cat_name.startswith(super_cat):
            normal_data.append(text)
        else:
            anomaly_data.append(text)
            
    # Split Train/Test
    split_idx = int(0.8 * len(normal_data))
    train_normal = normal_data[:split_idx]
    test_normal = normal_data[split_idx:]
    
    # Ratio 90% Normal / 10% Anomaly
    n_anomalies = int(len(test_normal) / 9)
    if n_anomalies > len(anomaly_data): n_anomalies = len(anomaly_data)
    test_anomalies = random.sample(anomaly_data, n_anomalies)
    
    test_texts = test_normal + test_anomalies
    test_labels = [0] * len(test_normal) + [1] * len(test_anomalies)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader = DataLoader(TextDataset(train_normal, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TextDataset(test_texts, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    
    # Train
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

    # Evaluation
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            z, _ = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            all_vecs.append(z)
            if i > 10: break
    model.c.data = torch.mean(torch.cat(all_vecs), dim=0, keepdim=True)
    scores_baseline = []
    scores_removed = []
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
            score_base = torch.sum((z - c) ** 2, dim=1).item()
            scores_baseline.append(score_base)
            
        embeddings_list = model.bert.embeddings.word_embeddings(input_ids)
        attributions = ig.attribute(inputs=embeddings_list, additional_forward_args=(mask), n_steps=100)
        token_importance = attributions.sum(dim=2)
        
        k = int(MAX_LEN * RATIONALE_PERCENTAGE)
        top_val, top_idx = torch.topk(token_importance, k, dim=1)
        top_indices = top_idx[0].tolist()
        
        modified_input_ids = input_ids.clone()
        removed_tokens = []
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        for idx in top_indices:
            if idx < len(all_tokens):
                removed_tokens.append(all_tokens[idx])
            modified_input_ids[0, idx] = 0 # PAD
            
        with torch.no_grad():
            z, c = model(input_ids=modified_input_ids, attention_mask=mask)
            score_removed = torch.sum((z - c) ** 2, dim=1).item()
            scores_removed.append(score_removed)
        
        # Save limited details
        is_anomaly = test_labels[i] == 1
        if i < 5 or (is_anomaly and i < (len(test_normal) + 5)):
            details_log.append([super_cat, run_id, i, "Anomaly" if is_anomaly else "Normal", score_base, score_removed, removed_tokens])

    auc_base = roc_auc_score(test_labels, scores_baseline)
    auc_removed = roc_auc_score(test_labels, scores_removed)
    
    return auc_base, auc_removed, details_log

def main():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_data = newsgroups.data
    all_targets = newsgroups.target
    target_names = newsgroups.target_names
    
    with open(FILE_SUMMARY, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SuperCategory", "AUC_Base_Mean", "AUC_Base_Std", "AUC_Removed_Mean", "AUC_Removed_Std", "Drop_Mean"])
        
    with open(FILE_DETAILS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["SuperCategory", "Run_ID", "Sample_ID", "Label", "Score_Original", "Score_Removed", "Removed_Rationales"])

    for super_cat in SUPER_CATEGORIES:
        print(f"Processing Super-Category: {super_cat}")
        base_aucs = []
        removed_aucs = []
        
        for run in range(NUM_RUNS):
            ab, ar, logs = train_and_evaluate(super_cat, run, all_data, all_targets, target_names)
            base_aucs.append(ab)
            removed_aucs.append(ar)
            
            with open(FILE_DETAILS, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(logs)
        
        mean_base = np.mean(base_aucs)
        std_base = np.std(base_aucs)
        mean_rem = np.mean(removed_aucs)
        std_rem = np.std(removed_aucs)
        drop = mean_base - mean_rem
        
        with open(FILE_SUMMARY, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([super_cat, mean_base, std_base, mean_rem, std_rem, drop])
            
        print(f"  > Done {super_cat}. Base: {mean_base:.3f}, Removed: {mean_rem:.3f}")

if __name__ == "__main__":
    main()
