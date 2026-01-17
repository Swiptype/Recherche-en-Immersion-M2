import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import os
from colorama import init, Fore, Style

# Initialisation couleurs
init(autoreset=True)

# --- CONFIGURATION ---
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5              # Un peu moins d'époques suffisent pour ce test
LEARNING_RATE = 2e-5
EMBED_DIM = 768
HIDDEN_DIM = 256
NU = 0.1
UNFREEZE_AFTER_EPOCH = 1

INPUT_RATIONALES = "rationales.txt"  # Le fichier généré par ton script précédent
OUTPUT_FLIPPING = "flipping_tokens.txt" # Le fichier de sortie de ce script

# --- 1. DATASET & MODELE (Identiques) ---
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

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        vec = outputs.last_hidden_state[:, 0, :] # Token [CLS]
        projected = self.projection(vec)
        score = torch.sum((projected - self.c) ** 2, dim=1)
        return score

# --- 2. FONCTIONS UTILITAIRES ---
def get_rationales_list(file_path):
    """Charge les mots toxiques depuis le fichier rationales.txt"""
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Erreur: Le fichier {file_path} n'existe pas. Lance le script précédent d'abord !{Style.RESET_ALL}")
        return []
    
    words = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            # Filtre basique pour éviter de tester des trucs inutiles
            if w and not w.startswith("-") and not w.startswith("Rat") and len(w) > 2:
                words.add(w)
    
    # On retourne une liste, limitée aux 50 premiers pour aller vite
    return list(words)[:50] 

def train_engine(normal_cat, device):
    """Entraîne le modèle rapidement pour avoir une base de détection"""
    print(f"\n{Fore.CYAN}--- Entraînement du modèle sur la classe : {normal_cat} ---{Style.RESET_ALL}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data_train = fetch_20newsgroups(subset='train', categories=[normal_cat], remove=('headers', 'footers', 'quotes')).data
    train_loader = DataLoader(TextDataset(data_train, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    
    model = TextAnomalyModel().to(device)
    
    # Init Center
    model.eval()
    vecs = []
    with torch.no_grad():
        for b in train_loader:
            inp, mask = b['input_ids'].to(device), b['attention_mask'].to(device)
            out = model.distilbert(inp, attention_mask=mask).last_hidden_state
            vecs.append(model.projection(out[:, 0, :]))
            if len(vecs) > 10: break
    model.c.data = torch.mean(torch.cat(vecs), dim=0, keepdim=True)

    # Train loop
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        if epoch == UNFREEZE_AFTER_EPOCH:
            for p in model.distilbert.parameters(): p.requires_grad = True
        model.train()
        total_loss = 0
        for b in train_loader:
            optimizer.zero_grad()
            scores = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    return model, tokenizer

# --- 3. EXPERIMENTATION : FLIPPING ---
def run_flipping_test(model, tokenizer, normal_cat, device):
    # 1. Charger les données de TEST
    test_data = fetch_20newsgroups(subset='test', categories=[normal_cat], remove=('headers', 'footers', 'quotes')).data
    # On prend 50 textes normaux au hasard pour le test
    test_samples = [t for t in test_data if len(t) > 50][:50]
    
    # 2. Calculer le SEUIL (Threshold)
    print(f"\n{Fore.CYAN}--- Calcul du seuil de normalité ---{Style.RESET_ALL}")
    model.eval()
    scores = []
    with torch.no_grad():
        for t in test_samples:
            inputs = tokenizer(t, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
            s = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device)).item()
            scores.append(s)
    
    threshold = np.quantile(scores, 1 - NU)
    print(f"Seuil établi à : {threshold:.4f}")

    # 3. Charger les Mots Toxiques
    dangerous_words = get_rationales_list(INPUT_RATIONALES)
    if not dangerous_words:
        print("Aucun mot chargé. Fin.")
        return

    print(f"Test d'injection avec {len(dangerous_words)} mots rationales...")
    
    # 4. Boucle d'injection
    if os.path.exists(OUTPUT_FLIPPING): os.remove(OUTPUT_FLIPPING)
    
    success_count = 0
    
    with open(OUTPUT_FLIPPING, "a", encoding="utf-8") as f:
        f.write(f"--- FLIPPING TOKENS ANALYSIS ---\n")
        f.write(f"Normal Class: {normal_cat}\n")
        f.write(f"Threshold: {threshold:.4f}\n\n")
        
        for i, text in enumerate(test_samples):
            # Vérif score initial
            inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
            with torch.no_grad():
                base_score = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device)).item()
            
            # Si le texte est déjà considéré comme une anomalie par erreur, on le saute
            if base_score > threshold: continue 

            # Essai d'injection
            for bad_word in dangerous_words:
                # Injection à la fin du texte
                corrupted_text = text + " " + bad_word
                
                inputs = tokenizer(corrupted_text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN).to(device)
                with torch.no_grad():
                    new_score = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device)).item()
                
                # SI LE SCORE EXPLOSE (Devient > Seuil)
                if new_score > threshold:
                    print(f"{Fore.GREEN}Flip réussi !{Style.RESET_ALL} '{bad_word}' a cassé le texte {i}.")
                    f.write(f"TEXT ID: {i}\n")
                    f.write(f"TOKEN: [{bad_word}]\n")
                    f.write(f"SCORE CHANGE: {base_score:.4f} -> {new_score:.4f} (Seuil: {threshold:.4f})\n")
                    f.write(f"CONTEXT: ...{text[-60:]} {bad_word}\n")
                    f.write("-" * 40 + "\n")
                    success_count += 1
                    break # On passe au texte suivant dès qu'on a trouvé une faille

    print(f"\n{Fore.YELLOW}--- RÉSULTATS ---{Style.RESET_ALL}")
    print(f"Nombre de textes 'normaux' convertis en anomalie : {success_count}")
    print(f"Détails sauvegardés dans : {OUTPUT_FLIPPING}")

# --- MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tu peux changer la catégorie ici pour tester
    # Exemple : on entraîne sur 'comp.graphics' (Informatique)
    # Les rationales (Jesus, God...) devraient faire exploser le score.
    target_class = 'comp.graphics' 
    
    # 1. Entraîner
    model, tokenizer = train_engine(target_class, device)
    
    # 2. Tester les rationales
    run_flipping_test(model, tokenizer, target_class, device)

if __name__ == "__main__":
    main()
