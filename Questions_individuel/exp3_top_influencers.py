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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LATENT_DIM = 64
MAX_LEN = 128
UNFREEZE_AFTER_EPOCH = 2
NUM_RUNS = 5

FILE_TOKENS = "results_exp3_tokens.csv"

# --- Classes ---
class DeepSVDD(nn.Module):
    def __init__(self):
        super(DeepSVDD, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.projection = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, LATENT_DIM))
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is not None:
             outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
             outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.last_hidden_state[:, 0, :])

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

def run_analysis(category, all_data, all_targets, target_names, tokenizer):
    target_idx = target_names.index(category)
    normal_data = [t for t, idx in zip(all_data, all_targets) if idx == target_idx]
    anomaly_data = [t for t, idx in zip(all_data, all_targets) if idx != target_idx]
    
    split_idx = int(0.8 * len(normal_data))
    train_normal = normal_data[:split_idx]
    
    # Analyze on a sample (e.g. 100 normal, 100 anomaly)
    analyze_normal = normal_data[split_idx:split_idx+100]
    analyze_anomalies = random.sample(anomaly_data, 100)
    
    train_loader = DataLoader(TextDataset(train_normal, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    loader_normal = DataLoader(TextDataset(analyze_normal, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    loader_anomaly = DataLoader(TextDataset(analyze_anomalies, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    
    model = DeepSVDD().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.eval()
    c = torch.zeros(LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        for batch in train_loader:
            c += model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE)).sum(dim=0)
            break 
    c /= BATCH_SIZE
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        if epoch >= UNFREEZE_AFTER_EPOCH:
             for param in model.bert.parameters(): param.requires_grad = True
        else:
             for param in model.bert.parameters(): param.requires_grad = False
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            loss = torch.mean(torch.sum((out - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()

    model.eval()
    def forward_score(inputs_embeds, attention_mask):
        return torch.sum((model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) - c) ** 2, dim=1)
    
    ig = IntegratedGradients(forward_score)
    
    local_impact_anomalies = defaultdict(float)
    local_impact_normals = defaultdict(float)
    
    # Analyze Anomalies
    for loader, store_dict in zip([loader_anomaly, loader_normal], [local_impact_anomalies, local_impact_normals]):
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            embeddings = model.bert.embeddings.word_embeddings(input_ids)
            
            attributions = ig.attribute(inputs=embeddings, additional_forward_args=(mask), n_steps=10)
            token_imp = attributions.sum(dim=2).squeeze(0)
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            for idx, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]']: continue
                store_dict[token] += token_imp[idx].item()
                
    return local_impact_anomalies, local_impact_normals

def main():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    all_data = newsgroups.data
    all_targets = newsgroups.target
    target_names = newsgroups.target_names
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    with open(FILE_TOKENS, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "TokenType", "Token", "AccumulatedScore"])

    for cat in target_names:
        print(f"Processing Category: {cat}")
        
        global_anom = defaultdict(float)
        global_norm = defaultdict(float)
        
        for run in range(NUM_RUNS):
            print(f"   Run {run+1}/{NUM_RUNS}")
            anom, norm = run_analysis(cat, all_data, all_targets, target_names, tokenizer)
            for k, v in anom.items(): global_anom[k] += v
            for k, v in norm.items(): global_norm[k] += v
            
        # Sort and Save Top 50
        sorted_anom = sorted(global_anom.items(), key=lambda x: x[1], reverse=True)[:50]
        sorted_norm = sorted(global_norm.items(), key=lambda x: x[1], reverse=False)[:50]
        
        with open(FILE_TOKENS, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for token, score in sorted_anom:
                writer.writerow([cat, "Anomaly_Driver", token, score/NUM_RUNS])

if __name__ == "__main__":
    main()