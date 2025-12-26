import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import numpy as np
import nltk
from nltk.corpus import reuters
import random

# --- Configuration Globale ---
# Choix du Dataset : '20NG' ou 'REUTERS'
DATASET_NAME = '20NG' 

# Choix du Modèle : 'AE' (Auto-Encodeur) ou 'CVDD' (Context Vector Data Description)
MODEL_TYPE = 'CVDD' 

# Paramètres
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5      # Augmenté pour meilleure convergence
LEARNING_RATE = 2e-5
EMBED_DIM = 768 # DistilBERT output
HIDDEN_DIM = 128 
NU = 0.1        # Quantile pour le seuil d'anomalie

# --- 1. Gestion des Données (Factory) ---

def load_reuters_data():
    """Charge et prépare Reuters-21578 (nécessite nltk.download('reuters'))."""
    try:
        nltk.data.find('corpora/reuters.zip')
    except LookupError:
        print("Téléchargement de Reuters via NLTK...")
        nltk.download('reuters')
        nltk.download('punkt')

    # On définit 'earn' comme classe normale (la plus fréquente dans Reuters)
    normal_category = 'earn'
    
    documents = reuters.fileids()
    train_docs = [d for d in documents if d.startswith('training/')]
    test_docs = [d for d in documents if d.startswith('test/')]

    # Fonction helper pour extraire le texte brut
    def get_text(doc_ids, label_filter=None, is_normal=True):
        texts = []
        for doc_id in doc_ids:
            categories = reuters.categories(doc_id)
            # Logique de filtrage In-Distribution vs Out-of-Distribution
            if is_normal:
                # On garde si ça appartient UNIQUEMENT ou PRINCIPALEMENT à la classe normale
                if normal_category in categories:
                    texts.append(reuters.raw(doc_id))
            else:
                # C'est une anomalie si ça n'appartient PAS à la classe normale
                if normal_category not in categories:
                    texts.append(reuters.raw(doc_id))
        return texts

    print(f"Dataset Reuters: Classe Normale = '{normal_category}'")
    train_texts = get_text(train_docs, is_normal=True)
    test_normal_texts = get_text(test_docs, is_normal=True)
    test_anomaly_texts = get_text(test_docs, is_normal=False) # Reste du monde

    return train_texts, test_normal_texts, test_anomaly_texts

def load_20ng_data():
    """Charge 20 Newsgroups."""
    normal_cat = 'sci.space'           #Remplacer par sci.med
    anomaly_cat = 'talk.religion.misc' #remplacer par sci.space ou comp.graphics
    print(f"Dataset 20NG: Normal='{normal_cat}', Anomaly='{anomaly_cat}'")
    
    train = fetch_20newsgroups(subset='train', categories=[normal_cat], remove=('headers', 'footers', 'quotes'))
    test_normal = fetch_20newsgroups(subset='test', categories=[normal_cat], remove=('headers', 'footers', 'quotes'))
    test_anomaly = fetch_20newsgroups(subset='test', categories=[anomaly_cat], remove=('headers', 'footers', 'quotes'))
    
    return train.data, test_normal.data, test_anomaly.data

def get_data(dataset_name):
    if dataset_name == 'REUTERS':
        try:
            return load_reuters_data()
        except Exception as e:
            print(f"Erreur chargement Reuters ({e}), fallback sur 20NG.")
            return load_20ng_data()
    else:
        return load_20ng_data()

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = [t for t in texts if len(t) > 10] # Filtre textes vides
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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- 2. Modèles (AE et CVDD) ---

class SelfAttention(nn.Module):
    """Couche d'attention simple pour CVDD."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.value = nn.Linear(embed_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        attention = torch.bmm(Q, K.transpose(1, 2)) / self.scale.to(x.device)
        attention = torch.softmax(attention, dim=-1)
        
        # Weighted sum
        context = torch.bmm(attention, V)
        return context, attention

class TextAnomalyModel(nn.Module):
    def __init__(self, mode='CVDD'):
        super(TextAnomalyModel, self).__init__()
        self.mode = mode
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Geler DistilBERT pour accélérer (optionnel, décommenter pour fine-tuning complet)
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False

        if self.mode == 'CVDD':
            # Architecture CVDD: Self-Attention -> Projection -> Hypersphere
            self.attention = SelfAttention(EMBED_DIM, HIDDEN_DIM)
            self.linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.c = nn.Parameter(torch.randn(1, HIDDEN_DIM), requires_grad=False)
            self.radius = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            
        elif self.mode == 'AE':
            # Architecture Auto-Encodeur
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(EMBED_DIM, 256),
                nn.ReLU(),
                nn.Linear(256, HIDDEN_DIM),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(HIDDEN_DIM, 256),
                nn.ReLU(),
                nn.Linear(256, EMBED_DIM)
            )
            # Seuil d'erreur de reconstruction
            self.threshold = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # (Batch, Seq, Dim)
        
        if self.mode == 'CVDD':
            # Appliquer attention
            context, _ = self.attention(sequence_output)
            # On prend le vecteur contextuel global (moyenne pondérée par attention ou CLS modifié)
            # Ici on fait un Mean Pooling sur la séquence pondérée
            vec = torch.mean(context, dim=1) 
            projected = self.linear(vec)
            
            # Score = Distance au centre c
            score = torch.sum((projected - self.c) ** 2, dim=1)
            return projected, score
            
        elif self.mode == 'AE':
            # Pour l'AE, on travaille souvent sur le token [CLS] ou la moyenne
            # Ici on utilise la moyenne des embeddings (Sentence Embedding simple)
            # Masking pour ne pas moyenner le padding
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask # (Batch, Dim)
            
            encoded = self.encoder(mean_embedding)
            decoded = self.decoder(encoded)
            
            # Score = Erreur de reconstruction (MSE) par échantillon
            score = torch.sum((mean_embedding - decoded) ** 2, dim=1)
            return decoded, score

    def get_embeddings(self, input_ids, attention_mask=None):
        return self.distilbert.embeddings(input_ids=input_ids)

# --- 3. Entraînement et Explication ---

def train(model, dataloader, device):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_mse = nn.MSELoss()
    
    # Init CVDD Center si nécessaire
    if model.mode == 'CVDD':
        print("Initialisation du centre CVDD...")
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            vec, _ = model(input_ids, mask)
            model.c.data = torch.mean(vec, dim=0, keepdim=True)

    model.train()
    print(f"Démarrage entraînement ({model.mode}) sur {DATASET_NAME}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            output, score = model(input_ids, mask)
            
            if model.mode == 'CVDD':
                # Loss = Distance moyenne au centre + Regularisation (optionnelle)
                loss = torch.mean(score)
            else: # AE
                # Loss = MSE entre entrée et sortie (calculé implicitement par le score qui est la SSE)
                # Mais pour la backprop stable, utilisons le module MSELoss sur les vecteurs
                # Le 'score' est déjà l'erreur carrée, donc mean(score) est la MSE
                loss = torch.mean(score)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

    # Définition du seuil (Radius pour CVDD, Threshold pour AE)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            _, score = model(input_ids, mask)
            all_scores.append(score)
    
    all_scores = torch.cat(all_scores)
    threshold_val = torch.quantile(all_scores, 1 - NU)
    
    if model.mode == 'CVDD':
        model.radius.data = threshold_val
        print(f"Seuil anomalie (Rayon R^2): {model.radius.item():.4f}")
    else:
        model.threshold.data = threshold_val
        print(f"Seuil anomalie (Reconstruction Error): {model.threshold.item():.4f}")


from captum.attr import IntegratedGradients

class Explainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        # CORRECTION : On utilise IntegratedGradients au lieu de LayerIntegratedGradients
        # car on va fournir directement les embeddings comme "input" à la fonction cible.
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, token_embeddings, attention_mask):
        # Cette fonction simule le réseau à partir des embeddings
        
        # Astuce technique pour DistilBERT : le masque d'attention doit être étendu
        # pour correspondre à la dimension [Batch, 1, 1, Seq] attendue par le transformer interne
        extended_mask = (1.0 - attention_mask) * -10000.0
        extended_mask = extended_mask.unsqueeze(1).unsqueeze(2)
        
        # On appelle directement le transformer (on saute la couche embeddings initiale du modèle)
        transformer_out = self.model.distilbert.transformer(
            x=token_embeddings, 
            attn_mask=extended_mask, 
            head_mask=[None] * self.model.distilbert.config.num_hidden_layers
        )[0]
        
        # Reste du réseau (CVDD ou AE)
        if self.model.mode == 'CVDD':
            context, _ = self.model.attention(transformer_out)
            vec = torch.mean(context, dim=1)
            projected = self.model.linear(vec)
            # Score = Distance au centre
            score = torch.sum((projected - self.model.c) ** 2, dim=1)
        else: # AE
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_out.size()).float()
            sum_embeddings = torch.sum(transformer_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_emb = sum_embeddings / sum_mask
            
            encoded = self.model.encoder(mean_emb)
            decoded = self.model.decoder(encoded)
            # Score = Erreur de reconstruction
            score = torch.sum((mean_emb - decoded) ** 2, dim=1)
            
        return score

    def explain(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        
        # On calcule les embeddings manuellement ici pour les donner comme entrée à IG
        input_embeddings = self.model.get_embeddings(input_ids)
        
        # Baseline (PAD)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
        baseline_embeddings = self.model.get_embeddings(baseline_ids)
        
        # Attributions
        # Note: on passe 'mask' dans additional_forward_args
        attrs = self.ig.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(mask),
            target=0, 
            n_steps=30, 
            internal_batch_size=1
        )
        
        # Score final pour affichage
        with torch.no_grad():
            _, score_val = self.model(input_ids, mask)
            
        # Aggrégation (Somme sur la dimension de l'embedding pour avoir un score par mot)
        word_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        
        return tokens, word_attrs, score_val.item()

# --- 4. Main ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {MODEL_TYPE} | Data: {DATASET_NAME}")
    
    # 1. Load Data
    train_txt, test_norm_txt, test_anom_txt = get_data(DATASET_NAME)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Augmenter artificiellement le train set pour l'exemple si trop petit
    if len(train_txt) < 1000:
        train_txt = train_txt * 2
        
    train_loader = DataLoader(TextDataset(train_txt, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model & Train
    model = TextAnomalyModel(mode=MODEL_TYPE).to(device)
    train(model, train_loader, device)
    
    # 3. Test & Explain
    explainer = Explainer(model, tokenizer)
    
    print("\n--- ANALYSE ---")
    
    # Test Normal
    t_norm = test_norm_txt[0]
    toks, attrs, score = explainer.explain(t_norm)
    threshold = model.radius.item() if MODEL_TYPE == 'CVDD' else model.threshold.item()
    print(f"\n[NORMAL] Score: {score:.4f} (Seuil: {threshold:.4f}) -> {'ANOMALIE' if score > threshold else 'OK'}")
    print(f"Texte début: {t_norm[:100]}...")
    
    # Test Anomalie
    t_anom = test_anom_txt[0]
    toks, attrs, score = explainer.explain(t_anom)
    print(f"\n[ANOMALIE] Score: {score:.4f} (Seuil: {threshold:.4f}) -> {'ANOMALIE' if score > threshold else 'OK'}")
    print(f"Texte début: {t_anom[:100]}...")
    
    # Affichage Rationale (Top 5 mots impactants)
    print("\nRationale de l'anomalie (Mots qui augmentent le plus le score):")
    indices = np.argsort(attrs)[-10:] # Top 10
    
    cleaned_rationale = []
    for i in reversed(indices):
        tok = toks[i]
        val = attrs[i]
        if tok not in ['[CLS]', '[SEP]', '[PAD]'] and not tok.startswith('##'):
            cleaned_rationale.append((tok, val))
            
    for tok, val in cleaned_rationale[:5]:
        print(f"  - {tok}: {val:.4f}")