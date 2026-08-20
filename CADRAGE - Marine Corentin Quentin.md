# 📋 Note de Cadrage · Projet Final MLDL M2
**Module :** Machine Learning & Deep Learning en Production (IPSSI · MIA5 26.1)  
**Projet :** BankRoute AI  
**Date :** Jeudi 20 août 2026 (Jalon 17h30)  
**Dépôt GitHub :** [https://github.com/Yamnyr/BankRoute-AI](https://github.com/Yamnyr/BankRoute-AI)

---

## 1. Secteur Choisi & Problématique Business

- **Secteur :** 🏦 **Banque & Relation Client**
- **Problématique Business :** Les services clients bancaires traitent des dizaines de milliers de sollicitations quotidiennes par chat et email. Une mauvaise affectation d'un ticket entraîne des délais de traitement, de l'insatisfaction client et un surcoût opérationnel.
- **Solution BankRoute AI :** Un service de Deep Learning déployé derrière une API FastAPI capable d'identifier instantanément l'intention du client parmi **77 motifs bancaires précis** et de router automatiquement le ticket vers le bon département opérationnel avec un score de confiance et les alternatives candidates.

---

## 2. Jeu de Données & Split

- **Dataset :** `banking77` (`mteb/banking77` sur Hugging Face)
- **Volumétrie :** 13 082 requêtes réelles étiquetées en anglais (9 993 train, 3 076 test indépendant).
- **Nombre de classes :** 77 intentions très fines couvrant tout le cycle bancaire (cartes perdues/volées, contestation de frais, virements, paiements déclinés, KYC, plafonds, etc.).
- **Split sans fuite :** Split Train (90%) / Validation (10%) sur le jeu d'entraînement, avec évaluation finale sur le Test set indépendant (`held-out`).
- **Routage Hiérarchique (Bonus) :** Agrégation des 77 intentions vers **7 départements opérationnels** :
  1. `Cards_Management` (Gestion & sécurité cartes)
  2. `Payments_Transfers` (Virements & prélèvements)
  3. `Account_Profile` (Profil, KYC & sécurité)
  4. `Cash_ATM` (Distributeurs & retraits espèces)
  5. `Top_Up_Deposits` (Recharges & dépôts)
  6. `Currency_Exchange` (Devises, taux de change & remboursements)
  7. `Digital_Services` (Apple/Google Pay & paiements connectés)

---

## 3. Modèle Envisagé & Architecture Technique

- **Architecture :** **DistilBERT** (`distilbert-base-uncased`, 66M paramètres) avec **LoRA (Low-Rank Adaptation - PEFT)**.
- **Paramétrage LoRA :** Matrices de décomposition de bas rang ($r=16, \alpha=32$, dropout $0.1$) injectées sur les projections d'attention `q_lin` et `v_lin`.
- **Efficacité paramétrique :** **~944k paramètres entraînables (~1.4% des poids)** pour un adaptateur ultra-léger de **~3.78 Mo** (contre ~256 Mo pour un checkpoint complet).
- **Baselines comparatives :**
  1. *Baseline 1 :* Classe Majoritaire (Dummy) $\rightarrow$ Accuracy attendue $\approx 1.30\%$.
  2. *Baseline 2 :* TF-IDF (unigrammes + bigrammes) + Régression Logistique $\rightarrow$ Accuracy $\approx 88.52\%$, Macro F1 $\approx 0.8856$.
  3. *Modèle Deep Learning :* DistilBERT + LoRA $\rightarrow$ Accuracy cible $\mathbf{> 91\%}$, Macro F1 $\mathbf{> 0.91}$.

---

## 4. Service de Production & API FastAPI

- **Framework :** FastAPI avec serveur Uvicorn asynchrone.
- **Gestion du Cycle de Vie :** Chargement unique du modèle et des mappings au démarrage via `@asynccontextmanager` (**lifespan**), avec préchauffage (*warmup*).
- **Typage & Validation Stricte :** Schémas Pydantic avec rejet systématique des requêtes invalides en **HTTP 422 Unprocessable Entity** (texte vide, types corrompus, `top_k` hors limites).
- **Endpoints :**
  - `GET /health` : Statut de santé et informations du modèle chargé.
  - `GET /intents` : Catalogue des 77 intentions et mapping des 7 départements.
  - `POST /predict` : Routage unitaire avec prédiction, score de confiance, département et Top-3.
  - `POST /predict/batch` : Routage par lot optimisé (un seul forward pass tensoriel).
  - `GET /docs` : Documentation Swagger UI interactive.

---

## 5. Plan GPU & Infrastructure

- **Matériel d'entraînement :** GPU dédié (NVIDIA RTX / Tesla T4), entraînement complet en **~80 secondes**.
- **Mode Fallback :** Sous-échantillonnage stratifié documenté dans `train.py` pour exécution CPU reproductible.
- **Latence d'inférence :** $p_{50} \approx 5.9\text{ ms}$ sur GPU / $\approx 16\text{ ms}$ sur CPU.
- **Conteneurisation :** `Dockerfile` optimisé avec wheels PyTorch CPU (`--extra-index-url`) et healthcheck automatisé.

---

## 6. Répartition des Rôles (« Qui Fait Quoi »)

| Membre | Rôles & Responsabilités Principales |
|---|---|
| **Corentin** | **Data Pipeline & Modélisation ML/DL :** Chargement du dataset `banking77`, baselines (Dummy, TF-IDF), configuration de l'adaptateur LoRA, calcul des métriques (Accuracy, Macro/Weighted F1, matrice de confusion) et notebook `BankRoute_AI_Training.ipynb`. |
| **Quentin** | **Architecture Logicielle & Service FastAPI :** Développement de `app.py`, gestionnaire de lifespan, validation stricte Pydantic (gestion des erreurs 422), enrichissement des prédictions (routage 7 départements) et endpoint batch. |
| **Marine** | **Tests, Déploiement & Narration :** Rédaction de la suite de tests automatisée `test_api.py` (httpx), benchmarking de latence, Dockerfile / `.dockerignore`, rédaction du `README.md` et préparation du scénario de soutenance. |

---

## 7. Livrables & Démo Préparée pour Vendredi

- [x] Notebook d'entraînement exécuté avec toutes les sorties visibles (`BankRoute_AI_Training.ipynb`)
- [x] Service de production (`app.py`) et client de test (`test_api.py`)
- [x] Artéfacts sérialisés (`model_artifact/`)
- [x] README exhaustif avec justifications architecturales
- [x] Dockerfile de production
- [x] Scénario de démo en direct : appel HTTP nominal, consultation `/docs`, et requête invalide rejetée en 422.
