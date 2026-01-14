import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import numpy as np

# Catégorie normale (In-Distribution) pour l'entraînement
NORMAL_CATEGORY = 'sci.space'
# Catégorie anormale (Out-of-Distribution) pour le test
ANOMALY_CATEGORY = 'talk.religion.misc'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
EMBED_DIM = 768  
HIDDEN_DIM = 64 
NU = 0.1

#Prepa données

def load_data():
    print(f"Chargement des données normales: {NORMAL_CATEGORY}")
    train_data = fetch_20newsgroups(
        subset='train',
        categories=[NORMAL_CATEGORY],
        remove=('headers', 'footers', 'quotes')
    )
    
    print(f"Chargement des données de test (normales): {NORMAL_CATEGORY}")
    test_normal_data = fetch_20newsgroups(
        subset='test',
        categories=[NORMAL_CATEGORY],
        remove=('headers', 'footers', 'quotes')
    )
    
    print(f"Chargement des données de test (anormales): {ANOMALY_CATEGORY}")
    test_anomaly_data = fetch_20newsgroups(
        subset='test',
        categories=[ANOMALY_CATEGORY],
        remove=('headers', 'footers', 'quotes')
    )
    
    return train_data.data, test_normal_data.data, test_anomaly_data.data

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

#Modèle (DistilBERT et Deep SVDD)

class DistilBertSVDD(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super(DistilBertSVDD, self).__init__()
        print("Initialisation du modèle DistilBERT...")
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.svdd_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )

        self.c = nn.Parameter(torch.randn(1, hidden_dim), requires_grad=False)
        self.radius = nn.Parameter(torch.tensor(0.0), requires_grad=False)


    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        latent_z = self.svdd_head(cls_embedding)
        dist = torch.sum((latent_z - self.c) ** 2, dim=1)
        
        return latent_z, dist

    def get_embeddings(self, input_ids, attention_mask):
        return self.distilbert.embeddings(input_ids=input_ids)

#Entrainement

def init_center_c(model, data_loader, device, eps=0.1):
    print("Initialisation du centre 'c'...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        latent_z, _ = model(input_ids, attention_mask)
        c = torch.mean(latent_z, dim=0, keepdim=True)

    c = c * (1 + eps)
    model.c.data = c
    print("Centre 'c' initialisé.")

def train_model(model, data_loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Début de l'entraînement sur {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            _, dist = model(input_ids, attention_mask)
            loss = torch.mean(dist)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(data_loader)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} terminée. Perte moyenne: {total_loss / len(data_loader):.4f}")

    print("Calcul du rayon R (score limite)...")
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, dist = model(input_ids, attention_mask)
            scores.append(dist.cpu())
    
    scores = torch.cat(scores)
    model.radius.data = torch.quantile(scores, 1 - NU)
    print(f"Rayon R^2 (score limite) fixé à: {model.radius.item():.4f}")


#Explicabilité (Integrated Gradients)

class TextAnomalyExplainer:  
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.ig = IntegratedGradients(self.model_forward_wrapper)

    def model_forward_wrapper(self, token_embeddings, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)

        transformer_outputs = self.model.distilbert.transformer(
            x=token_embeddings, 
            attn_mask=extended_attention_mask,
            head_mask=[None] * self.model.distilbert.config.num_hidden_layers
        )
        hidden_state = transformer_outputs[0] 
        cls_embedding = hidden_state[:, 0]
        latent_z = self.model.svdd_head(cls_embedding)
        dist = torch.sum((latent_z - self.model.c) ** 2, dim=1)
        
        return dist

    def create_baseline(self, input_ids):
        pad_token_id = self.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_token_id)
        baseline_embeddings = self.model.get_embeddings(baseline_ids, attention_mask=None)
        return baseline_embeddings

    def get_attributions(self, text):
        self.model.eval()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        baseline_embeddings = self.create_baseline(input_ids)

        with torch.no_grad():
            _, score = self.model(input_ids, attention_mask)

        attributions = self.ig.attribute(
            inputs=self.model.get_embeddings(input_ids, attention_mask=None),
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask),
            n_steps=50,
            internal_batch_size=1 
        )

        token_attributions = attributions.sum(dim=-1).squeeze(0)
        token_attributions = token_attributions / torch.norm(token_attributions)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        return tokens, token_attributions.cpu().detach().numpy(), score.item()

    def extract_rationale(self, tokens, attributions, num_tokens=10):
        special_tokens = [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]
        valid_indices = [i for i, token in enumerate(tokens) if token not in special_tokens and token.strip()]
        
        valid_tokens = [tokens[i] for i in valid_indices]
        valid_attributions = attributions[valid_indices]

        top_k_indices = np.argsort(valid_attributions)[-num_tokens:]
        
        rationale = [(valid_tokens[i], valid_attributions[i]) for i in top_k_indices]
        rationale.sort(key=lambda x: x[1], reverse=True)
        
        return rationale

#Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_texts, test_normal_texts, test_anomaly_texts = load_data()

    train_texts = train_texts
    test_normal_texts = test_normal_texts[:50]
    test_anomaly_texts = test_anomaly_texts[:50]

    train_dataset = TextDataset(train_texts, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DistilBertSVDD().to(device)

    init_center_c(model, train_loader, device)

    train_model(model, train_loader, device)

    explainer = TextAnomalyExplainer(model, tokenizer)
    
    print("\n" + "="*50)
    print("--- Test sur un exemple NORMAL (In-Distribution) ---")
    print(f"Texte: {test_normal_texts[0][:200]}...")
    
    tokens, attrs, score = explainer.get_attributions(test_normal_texts[0])
    rationale = explainer.extract_rationale(tokens, attrs, num_tokens=5)
    
    print(f"\nScore d'anomalie: {score:.4f} (Attendu: faible, < {model.radius.item():.4f})")
    print("Rationale (mots contribuant le plus au score):")
    for token, attr in rationale:
        print(f"  - {token}: {attr:.4f}")

    print("\n" + "="*50)
    print("--- Test sur un exemple ANORMAL (Out-of-Distribution) ---")
    print(f"Texte: {test_anomaly_texts[0][:200]}...")

    tokens, attrs, score = explainer.get_attributions(test_anomaly_texts[0])
    rationale = explainer.extract_rationale(tokens, attrs, num_tokens=5)

    print(f"\nScore d'anomalie: {score:.4f} (Attendu: élevé, > {model.radius.item():.4f})")
    print("Rationale (mots contribuant le plus au score):")
    for token, attr in rationale:
        print(f"  - {token.replace('##', '')}: {attr:.4f}")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    main()