import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.datasets import fetch_20newsgroups
from captum.attr import FeatureAblation
import numpy as np

# --- Configuration Globale ---
# Nom du dataset (informatif)
DATASET_NAME = '20NG'          # Ou Reuters-21578
# Nom du modèle (utilisé dans les logs)
MODEL_TYPE = 'CVDD'            # Ou AE pour Autoencoder

# --- Hyperparamètres du Modèle et de l'Entraînement ---
MAX_LEN = 128           # Longueur maximale des séquences pour le tokenizer
BATCH_SIZE = 16         # Taille des lots pour l'entraînement
EPOCHS = 12             # Nombre total d'époques d'entraînement
LEARNING_RATE = 5e-5    # Taux d'apprentissage pour l'optimiseur AdamW
EMBED_DIM = 768         # Dimension des embeddings de DistilBERT
HIDDEN_DIM = 256        # Dimension des couches cachées dans notre modèle
NU = 0.1                # Hyperparamètre de SVDD, proportion attendue d'anomalies

# --- Paramètres Anti-Collapse ---
# Le but est d'éviter que le modèle ne produise des embeddings identiques pour tous les inputs.
LAMBDA_REG = 1.0       # Poids de la pénalité de variance (pour encourager la diversité des embeddings)
MIN_RADIUS = 1e-4      # Rayon minimal de la sphère pour éviter un seuil nul
UNFREEZE_AFTER_EPOCH = 3  # Époque à partir de laquelle on dégèle des couches de DistilBERT pour le fine-tuning

# --- 1. Chargement et Préparation des Données ---
def load_20ng_data():
    """Charge les données du dataset 20 Newsgroups."""
    normal_cat = 'sci.med'    # Catégorie considérée comme normale
    anomaly_cat = 'sci.space' # Catégorie considérée comme anormale
    print(f"Dataset: Normal='{normal_cat}' vs Anomaly='{anomaly_cat}'")

    # On charge uniquement les données normales pour l'entraînement
    train = fetch_20newsgroups(subset='train', categories=[normal_cat], remove=('headers', 'footers', 'quotes'))
    # On charge les données de test pour les deux catégories
    test_normal = fetch_20newsgroups(subset='test', categories=[normal_cat], remove=('headers', 'footers', 'quotes'))
    test_anomaly = fetch_20newsgroups(subset='test', categories=[anomaly_cat], remove=('headers', 'footers', 'quotes'))

    return train.data, test_normal.data, test_anomaly.data

class TextDataset(Dataset):
    """Classe Dataset pour PyTorch, pour tokeniser les textes à la volée."""
    def __init__(self, texts, tokenizer, max_len):
        # On filtre les textes trop courts qui peuvent nuire à la stabilité de l'entraînement
        self.texts = [t for t in texts if isinstance(t, str) and len(t) > 50]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Retourne le nombre de textes dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Récupère et tokenise un texte par son index."""
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # Ajoute [CLS] et [SEP]
            max_length=self.max_len,      # Tronque ou padde à la longueur max
            padding='max_length',         # Padding jusqu'à max_len
            truncation=True,              # Tronque si plus long
            return_attention_mask=True,   # Retourne le masque d'attention
            return_tensors='pt'           # Retourne des tenseurs PyTorch
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- 2. Définition du Modèle ---
class SelfAttention(nn.Module):
    """Couche d'auto-attention simple pour agréger les embeddings de tokens."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, hidden_dim)
        self.key = nn.Linear(embed_dim, hidden_dim)
        self.value = nn.Linear(embed_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, x):
        """Calcule la sortie de l'attention."""
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Calcul des scores d'attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.scale.to(x.device) + 1e-8)
        attention = torch.softmax(scores, dim=-1)
        # Application des scores sur la matrice Value
        return torch.bmm(attention, V)

class TextAnomalyModel(nn.Module):
    """Le modèle principal qui combine DistilBERT, Attention et une tête de projection."""
    def __init__(self):
        super().__init__()
        # Modèle de base DistilBERT pré-entraîné
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # On gèle initialement tous les paramètres de DistilBERT
        # pour stabiliser l'entraînement des couches ajoutées.
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Couche d'attention pour agréger les embeddings de sortie de DistilBERT
        self.attention = SelfAttention(EMBED_DIM, HIDDEN_DIM)

        # Tête de projection qui réduit la dimension et normalise les vecteurs
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM)
        )

        # Le centre 'c' de la sphère SVDD. Non entraînable, il est fixé après initialisation.
        self.c = nn.Parameter(torch.randn(1, HIDDEN_DIM), requires_grad=False)
        # Le rayon de la sphère (seuil), non entraînable, fixé après l'entraînement.
        self.radius = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, return_embedding=False):
        """Passe avant du modèle. Calcule le score d'anomalie."""
        # Assure qu'on ne passe pas à la fois des IDs et des embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Vous ne pouvez pas spécifier à la fois input_ids et inputs_embeds")

        # Passe dans DistilBERT
        out = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        ).last_hidden_state

        # Agrégation des embeddings de tokens via l'attention
        context = self.attention(out)
        # Pooling pour obtenir un vecteur de phrase
        vec = torch.mean(context, dim=1)
        # Projection dans l'espace de la SVDD
        projected = self.projection(vec)
        # Le score est la distance euclidienne carrée au centre 'c'
        score = torch.sum((projected - self.c) ** 2, dim=1)

        if return_embedding:
            return score, projected
        else:
            return score

    def get_embeddings(self, input_ids):
        """Récupère les embeddings d'entrée (utilisé par l'explainer)."""
        return self.distilbert.embeddings(input_ids=input_ids)

# --- 3. Fonction de Perte ---
def compute_loss_with_regularization(model, scores, projected_embeddings):
    """
    Calcule la perte SVDD avec une pénalité pour maximiser la variance.
    Loss = SVDD_loss + λ * variance_penalty
    """
    # Perte SVDD : on minimise la distance moyenne au centre
    svdd_loss = torch.mean(scores)

    # Pénalité de variance : on veut maximiser la variance des embeddings projetés
    # pour éviter que le modèle ne produise la même sortie pour tout.
    embedding_var = torch.var(projected_embeddings, dim=0).mean()
    variance_penalty = -embedding_var  # Négatif car on maximise

    # Combinaison des deux pertes
    total_loss = svdd_loss + LAMBDA_REG * variance_penalty

    return total_loss, svdd_loss.item(), embedding_var.item()

# --- 4. Explicabilité avec Captum ---
class Explainer:
    """Wrapper pour les méthodes d'explicabilité de Captum."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # On utilise FeatureAblation qui fonctionne par perturbation.
        # On définit une fonction forward que l'ablator peut appeler.
        def forward_func(input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        self.ablator = FeatureAblation(forward_func)

    def explain(self, text):
        """Calcule les attributions des mots pour un texte donné."""
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)

        # La baseline pour l'ablation est un token neutre et fréquent comme "the".
        # Remplacer un mot par "the" est une meilleure baseline que [PAD] ou [UNK].
        the_id = self.tokenizer.convert_tokens_to_ids('the')
        baseline_ids = torch.full_like(input_ids, fill_value=the_id)
        
        # Calcule l'attribution : score_original - score_après_ablation
        attrs = self.ablator.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(mask,)
        )

        with torch.no_grad():
            score_val = self.model(input_ids=input_ids, attention_mask=mask).item()

        # Récupération des attributions et des tokens
        word_attrs = attrs.squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        return tokens, word_attrs, score_val

# --- 5. Boucle Principale ---
def run():
    """Exécute toutes les étapes : chargement, entraînement, test, et explication."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Chargement des données
    train_txt, test_norm, test_anom = load_20ng_data()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = TextDataset(train_txt, tokenizer, MAX_LEN)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialisation du modèle
    model = TextAnomalyModel().to(device)

    # 3. Initialisation du centre 'c'
    # On calcule le centre initial comme la moyenne des embeddings (non entraînés)
    # sur un sous-ensemble des données d'entraînement.
    print("Initialisation du Centre c avec variance (sur embeddings projetés)...")
    model.eval()
    all_vecs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:  # Limite pour accélérer l'initialisation
                break
            inp = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            _, emb = model(inp, mask, return_embedding=True)
            all_vecs.append(emb)
        if len(all_vecs) == 0:
            raise RuntimeError("Aucune donnée d'initialisation (dataset trop petit ou filtrage trop strict).")
        all_embeddings = torch.cat(all_vecs, dim=0)
        model.c.data = torch.mean(all_embeddings, dim=0, keepdim=True)
        init_var = torch.var(all_embeddings, dim=0).mean().item()
        print(f"Centre initialisé - Norme: {torch.norm(model.c).item():.4f}, Variance initiale: {init_var:.6f}")

    # 4. Configuration de l'optimiseur et du scheduler
    optimizer = optim.AdamW([
        {'params': [p for p in model.distilbert.parameters() if p.requires_grad], 'lr': LEARNING_RATE},
        {'params': model.attention.parameters(), 'lr': LEARNING_RATE * 5},
        {'params': model.projection.parameters(), 'lr': LEARNING_RATE * 5}
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Boucle d'entraînement
    model.train()
    print(f"\nEntraînement avec λ_reg={LAMBDA_REG}...")
    for epoch in range(EPOCHS):
        total_loss, total_svdd, total_var, nb_batches = 0.0, 0.0, 0.0, 0

        # Dégel progressif des couches de DistilBERT
        if epoch == UNFREEZE_AFTER_EPOCH:
            try:
                n_layers = len(model.distilbert.transformer.layer)
                # On dégèle les 2 dernières couches
                for lyr in model.distilbert.transformer.layer[n_layers - 2: n_layers]:
                    for p in lyr.parameters():
                        p.requires_grad = True
                # On ajoute les nouveaux paramètres à l'optimiseur sans le réinitialiser
                new_params = [p for lyr in model.distilbert.transformer.layer[n_layers - 2: n_layers] for p in lyr.parameters()]
                optimizer.add_param_group({'params': new_params, 'lr': LEARNING_RATE})
                print(f"Epoch {epoch+1}: Dégel des 2 dernières couches de DistilBERT pour fine-tuning.")
            except Exception as e:
                print(f"Impossible de dégel des couches DistilBERT: {e}")

        for batch in loader:
            optimizer.zero_grad()
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)

            # Passe avant
            score, embeddings = model(inp, mask, return_embedding=True)

            # Calcul de la perte
            loss, svdd_loss, var = compute_loss_with_regularization(model, score, embeddings)

            # Rétropropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_svdd += svdd_loss
            total_var += var
            nb_batches += 1

        if nb_batches == 0:
            raise RuntimeError("Aucun batch traité pendant l'entraînement.")
        
        # Mise à jour du taux d'apprentissage
        scheduler.step()

        avg_loss = total_loss / nb_batches
        avg_svdd = total_svdd / nb_batches
        avg_var = total_var / nb_batches
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} ({MODEL_TYPE}={avg_svdd:.4f}, Var={avg_var:.6f})")

    # 6. Calcul du seuil d'anomalie
    # Le seuil est le quantile (1 - nu) des scores sur les données d'entraînement.
    model.eval()
    scores = []
    print("\nCalcul du seuil (sur le training set)...")
    with torch.no_grad():
        for batch in loader:
            inp, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            sc = model(inp, mask)
            scores.append(sc)
    all_scores = torch.cat(scores)
    threshold = torch.quantile(all_scores, 1 - NU).item()
    threshold = max(threshold, MIN_RADIUS)  # Assure un seuil minimal
    model.radius.data = torch.tensor(threshold)
    print(f"Seuil fixé à: {threshold:.6f}")
    print(f"Stats - Min: {all_scores.min().item():.6f}, Max: {all_scores.max().item():.6f}, Mean: {all_scores.mean().item():.6f}")

    # 7. Évaluation sur les données de test
    explainer = Explainer(model, tokenizer)

    print("\n" + "=" * 50)
    print("--- TESTS NORMAUX (Medical) ---")
    normal_scores = []
    for i in range(min(10, len(test_norm))):
        t_norm = test_norm[i]
        _, _, score = explainer.explain(t_norm)
        normal_scores.append(score)
        status = '✓ NORMAL' if score <= threshold else '✗ ANOMALIE'
        print(f"Test {i+1}: Score={score:.6f} -> {status}")
    normal_correct = sum(1 for s in normal_scores if s <= threshold)
    print(f"\n✓ Précision Normal: {normal_correct}/{len(normal_scores)} ({100 * normal_correct / len(normal_scores):.1f}%)")

    print("\n" + "=" * 50)
    print("--- TESTS ANOMALIES (Espace) ---")
    anomaly_scores = []
    for i in range(min(10, len(test_anom))):
        t_anom = test_anom[i]
        _, _, score = explainer.explain(t_anom)
        anomaly_scores.append(score)
        status = '✓ ANOMALIE' if score > threshold else '✗ NORMAL'
        print(f"Test {i+1}: Score={score:.6f} -> {status}")
    anomaly_correct = sum(1 for s in anomaly_scores if s > threshold)
    print(f"\n✓ Précision Anomalie: {anomaly_correct}/{len(anomaly_scores)} ({100 * anomaly_correct / len(anomaly_scores):.1f}%)")

    # 8. Statistiques et Rationale
    print("\n" + "=" * 50)
    print("--- STATISTIQUES GLOBALES ---")
    print(f"Seuil: {threshold:.6f}")
    print(f"Scores Normaux  - Mean: {np.mean(normal_scores):.6f}")
    print(f"Scores Anomalies - Mean: {np.mean(anomaly_scores):.6f}")
    print(f"Séparation (diff des moyennes): {abs(np.mean(anomaly_scores) - np.mean(normal_scores)):.6f}")

    print("\n" + "=" * 50)
    print("--- RATIONALE (Anomalie la plus claire) ---")
    if len(anomaly_scores) == 0:
        print("Aucun exemple anomalie pour calculer la rationale.")
        return

    # On choisit l'anomalie avec le score le plus élevé pour l'explication
    best_anom_idx = int(np.argmax(anomaly_scores))
    toks, attrs, score = explainer.explain(test_anom[best_anom_idx])

    print(f"Texte: {test_anom[best_anom_idx][:200]}...")
    print(f"\nScore: {score:.6f} (Seuil: {threshold:.6f})")
    print("\nMots contribuant le plus à l'anomalie (attributions positives):")

    # On filtre pour ne garder que les attributions positives des vrais tokens
    valid_pairs = [(toks[i], float(attrs[i])) for i in range(len(toks))
                   if toks[i] not in ['[CLS]', '[SEP]', '[PAD]'] and attrs[i] > 0]
    valid_pairs.sort(key=lambda x: x[1], reverse=True)

    if not valid_pairs:
        print("Aucune attribution positive trouvée.")

    for token, attr in valid_pairs[:30]:
        print(f"  {token:15s}: +{attr:.6f}")

if __name__ == "__main__":
    run()
