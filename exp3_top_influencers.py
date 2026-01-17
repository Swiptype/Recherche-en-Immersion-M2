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

# --- Configuration ---
CATEGORY_NORMAL = 'comp.graphics'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
LATENT_DIM = 64
MAX_LEN = 128
UNFREEZE_AFTER_EPOCH = 2

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

def run_experiment():
    print(f"--- Experiment 3: Identifying Flipping Tokens (Category: {CATEGORY_NORMAL}) ---")
    
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
    target_idx = newsgroups.target_names.index(CATEGORY_NORMAL)
    
    normal_data = data[data['target'] == target_idx]['text'].tolist()
    anomaly_data = data[data['target'] != target_idx]['text'].tolist()
    
    split_idx = int(0.8 * len(normal_data))
    train_normal = normal_data[:split_idx]
    
    analyze_normal = normal_data[split_idx:split_idx+200]
    analyze_anomalies = random.sample(anomaly_data, 200)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader = DataLoader(TextDataset(train_normal, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    loader_normal = DataLoader(TextDataset(analyze_normal, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    loader_anomaly = DataLoader(TextDataset(analyze_anomalies, tokenizer, MAX_LEN), batch_size=1, shuffle=False)
    
    model = DeepSVDD().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.eval()
    c = torch.zeros(LATENT_DIM).to(device)
    with torch.no_grad():
        for batch in train_loader:
            c += model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).sum(dim=0)
            break 
    c /= BATCH_SIZE
    
    print("Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        if epoch >= UNFREEZE_AFTER_EPOCH:
             for param in model.bert.parameters(): param.requires_grad = True
        else:
             for param in model.bert.parameters(): param.requires_grad = False
        for batch in train_loader:
            optimizer.zero_grad()
            loss = torch.mean(torch.sum((model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)) - c) ** 2, dim=1))
            loss.backward()
            optimizer.step()

    word_impact_anomalies = defaultdict(float)
    word_impact_normals = defaultdict(float)
    
    def forward_score(inputs_embeds, attention_mask):
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return torch.sum((out - c) ** 2, dim=1)
    
    ig_manual = IntegratedGradients(forward_score)
    
    print("Analyzing...")
    model.eval()
    for loader, impact_dict in zip([loader_anomaly, loader_normal], [word_impact_anomalies, word_impact_normals]):
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            embeddings_list = model.bert.embeddings.word_embeddings(input_ids)
            
            attributions = ig_manual.attribute(
                inputs=embeddings_list,
                additional_forward_args=(mask),
                n_steps=10
            )
            token_importance = attributions.sum(dim=2).squeeze(0)
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            for idx, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]']: continue
                impact_dict[token] += token_importance[idx].item()

    print("\nTop words pushing towards ANOMALY (High attribution):")
    sorted_anom = sorted(word_impact_anomalies.items(), key=lambda x: x[1], reverse=True)
    for w, s in sorted_anom[:20]:
        print(f"{w}: {s:.2f}")

if __name__ == "__main__":
    run_experiment()