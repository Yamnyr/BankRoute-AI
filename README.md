# 🏦 BankRoute AI · Système Intelligent de Routage de Support Bancaire
> **Projet Final MLDL M2 (IPSSI · MIA5 26.1)**  
> *Module Machine Learning & Deep Learning en Production*

---

## 📑 Sommaire
1. [Vue d'Ensemble & Contexte Métier](#1-vue-densemble--contexte-métier)
2. [Jeu de Données & Préparation (`banking77`)](#2-jeu-de-données--préparation-banking77)
3. [Architecture & Justification des Choix Techniques](#3-architecture--justification-des-choix-techniques)
4. [Mesure Honnête & Comparaison des Performances](#4-mesure-honnête--comparaison-des-performances)
5. [Infrastructure GPU & Dimensionnement](#5-infrastructure-gpu--dimensionnement)
6. [Architecture du Service de Production FastAPI (`app.py`)](#6-architecture-du-service-de-production-fastapi-apppy)
7. [Client de Test & Validation HTTP (`test_api.py`)](#7-client-de-test--validation-http-test_apipy)
8. [Guide de Lancement Rapide (Local & Docker)](#8-guide-de-lancement-rapide-local--docker)
9. [Bonus Revendiqués](#9-bonus-revendiqués)
10. [Limites & « Avec une Semaine de Plus »](#10-limites--avec-une-semaine-de-plus)

---

## 1. Vue d'Ensemble & Contexte Métier

Dans le secteur bancaire et fintech, le service client reçoit quotidiennement des dizaines de milliers de sollicitations par chat, email et application mobile. Une mauvaise orientation d'un ticket (ex: envoyer une contestation de fraude vers le service des relevés de compte) génère de la latence, de l'insatisfaction client et un surcoût opérationnel majeur.

**BankRoute AI** résout ce défi en déployant un modèle de Deep Learning capable de :
1. Classifier instantanément la requête client parmi **77 intentions bancaires fines**.
2. Calculer un **score de confiance probabiliste** (Softmax) et identifier les **Top-3 alternatives**.
3. Router automatiquement le ticket vers l'un des **7 départements opérationnels** de la banque.
4. Répondre en temps réel (**p50 ≈ 6.7 ms sur GPU**, mesuré en conditions réelles via `test_api.py`) derrière une **API FastAPI typée et sécurisée**.

```mermaid
flowchart LR
    A[Message Client\n'I lost my card abroad'] --> B[FastAPI /predict\nPydantic 422 Check]
    B --> C[DistilBERT Fine-Tuned\n77 Intentions]
    C --> D[Intention : lost_or_stolen_card\nConfiance : 16.5%]
    D --> E[Routage : Cards_Management\n'Blocage & Sécurité Cartes']
```

---

## 2. Jeu de Données & Préparation (`banking77`)

### 2.1 Pourquoi ce dataset ?
Nous avons retenu le jeu de données de référence **`banking77`** (Casanueva et al., PolyAI / Hugging Face). 
- **Taille :** 13 082 requêtes réelles étiquetées en anglais.
- **Granularité :** 77 intentions couvrant l'intégralité du cycle bancaire (gestion de carte, virements, paiements déclinés, plafonds, frais, devises, vérification d'identité KYC).
- **Défi :** Les intentions sont sémantiquement très proches (ex: `card_not_working` vs `contactless_not_working`, ou `exchange_rate` vs `exchange_charge`), ce qui rend les approches par mots-clés insuffisantes.

### 2.2 Split et Prévention des Fuites
- **Train brut :** 9 993 exemples
- **Test indépendant (held-out) :** 3 076 exemples
- **Split Train / Validation interne :** Répartition 90% train / 10% val sans chevauchement.
- **Sous-échantillonnage maîtrisé (Règle 4 de l'énoncé) :** Pour les cycles d'itération et tests CPU, un sous-échantillonnage stratifié représentatif (3 500 train, 500 val, 1 000 test) est appliqué afin de garantir un temps de calcul maîtrisé tout en conservant l'équilibre des 77 classes.

---

## 3. Architecture & Justification des Choix Techniques

### 3.1 Pourquoi DistilBERT avec LoRA (PEFT) ?

Nous utilisons la méthode **LoRA (Low-Rank Adaptation - PEFT)** sur l'encodeur `distilbert-base-uncased` :
- **Principe :** Gel des poids pré-entraînés du Transformer et injection de matrices de décomposition de bas rang ($A$ et $B$ avec $r=16, \alpha=32$) sur les projections d'attention (`q_lin`, `v_lin`).
- **Efficacité paramétrique :** **944 717 paramètres entraînables** sur 67 957 402 (**~1.4% des paramètres totaux**, mesuré via `model.print_trainable_parameters()`).
- **Gain de stockage :** L'adaptateur pèse **~3.78 Mo** (`adapter_model.safetensors`) contre ~268 Mo pour un checkpoint complet, facilitant le déploiement et le versioning multi-tâches.

| Modèle / Approche | Paramètres Entraînables | Accuracy (test, mesurée) | Poids Artéfact | Verdict |
|---|---|---|---|---|
| **Classe Majoritaire (Dummy)** | 0 | 1.30% | 0 Ko | ❌ Baseline naïve |
| **TF-IDF + Régression Logistique** | ~770 000 | 88.52% | ~15 Mo | ⚠️ Solide mais aveugle à la sémantique fine |
| **DistilBERT Full Fine-Tuning** | 67 012 685 (100%) | 91.84% | ~256 Mo | ⚠️ +0.81 pt vs LoRA, pour ~68× plus de poids à versionner |
| **DistilBERT + LoRA (BankRoute AI)** | **944 717 (~1.4%)** | **91.03%** | **~3.78 Mo** | **Retenu : quasi la même accuracy pour une fraction du coût de stockage/déploiement** |

*Full Fine-Tuning mesuré pour de vrai (5 epochs, même run GPU) : 91.84% accuracy / 0.9184 macro F1 / 0.9184 weighted F1, entraîné en 54.7 s. Il bat légèrement LoRA (+0.81 pt d'accuracy) mais produit un checkpoint ~68× plus lourd (256 Mo vs 3.78 Mo) pour un gain marginal — c'est précisément l'arbitrage qui justifie le choix de LoRA pour la production. Le déploiement utilise exclusivement la variante LoRA.*

---

## 4. Mesure Honnête & Comparaison des Performances

Les modèles ont été évalués sur le jeu de test indépendant (`held-out test set`) avec trois métriques complémentaires :
1. **Accuracy globale** : pourcentage de prédictions exactement correctes.
2. **Macro F1-Score** : moyenne non pondérée du F1 sur les 77 classes (sanctionne lourdement les échecs sur les classes minoritaires).
3. **Weighted F1-Score** : F1 pondéré par le support de chaque classe.

### Tableau Comparatif des Résultats

Chiffres issus de `model_artifact/metrics_summary.json`, généré automatiquement par `train.py` à chaque entraînement (aucune valeur saisie à la main).

| Modèle | Accuracy | Macro F1 | Weighted F1 | Gain vs Baseline |
|---|---|---|---|---|
| **Baseline 1 : Classe Majoritaire** | 1.30 % | 0.0003 | 0.0003 | Référence |
| **Baseline 2 : TF-IDF + LogReg** | 88.52 % | 0.8856 | 0.8857 | +87.22 % |
| **BankRoute AI (DistilBERT + LoRA)** | **91.03 %** | **0.9103** | **0.9103** | **+89.73 % vs majoritaire, +2.50 % vs TF-IDF** |

Latence : 0.35 ms en moyenne pour l'évaluation batchée sur GPU pendant l'entraînement (`avg_latency_ms` du run), contre **p50 ≈ 6.73 ms / p95 ≈ 9.34 ms** mesurés en conditions réelles via `test_api.py` (requête HTTP individuelle, incluant tokenization + sérialisation JSON + aller-retour réseau local).

### Analyse des Limites & Confusions Résiduelles
- **Erreurs observées :** La suite `test_api.py` a elle-même capturé un exemple réel de confusion sur les 6 cas métiers testés (15/16 tests passés, 93.8%) : *"Why is there an unexpected extra fee charged on my statement for currency exchange?"* a été routé vers `Cards_Management` (intention prédite : `card_payment_wrong_exchange_rate`) au lieu du `Currency_Exchange` attendu — les deux intentions concernent des frais liés au change et sont sémantiquement très proches.
- **Limite du routage hiérarchique :** Mesure systématique effectuée sur l'ensemble du test set (77 classes × 3076 exemples), en agrégeant les 77 intentions vers les 7 départements via `DEPARTMENT_MAPPING` puis en comparant département prédit vs département réel (`train.py::train_deep_learning_model`, matrice complète exportée dans `model_artifact/confusion_matrices.json`) : **accuracy département = 95.51 %**, contre 91.03 % au niveau intention. Le "sauvetage" par la hiérarchie est donc réel dans la majorité des cas, mais pas systématique : l'exemple ci-dessus montre que la confusion peut aussi traverser la frontière département sur ~4.5 % des cas de test.

---

## 5. Infrastructure GPU & Dimensionnement

- **Matériel utilisé pour ce run :** GPU NVIDIA GeForce RTX 5070 (poste local) — entraînement complet (dataset entier, 10 epochs, LoRA) en **88.41 s**. Fallback CPU disponible via un sous-échantillonnage documenté dans `train.py` (`load_and_prepare_data`).
- **Paramètres entraînables :** 944 717 (LoRA + `classifier`/`pre_classifier` non gelés) sur 67 957 402 au total.
- **Budget temps de réponse mesuré :** p50 ≈ 6.73 ms / p95 ≈ 9.34 ms par requête individuelle via HTTP (`test_api.py`, GPU).

---

## 6. Architecture du Service de Production FastAPI (`app.py`)

### 6.1 Gestion du Cycle de Vie (`@asynccontextmanager lifespan`)
Le modèle, le tokenizer et les dictionnaires de correspondance sont chargés en mémoire vive **une seule et unique fois au démarrage** de FastAPI. Chaque appel HTTP réutilise les tenseurs déjà alloués, évitant tout rechargement disque coûteux.

### 6.2 Validation Pydantic Stricte (Gestion des Erreurs 422)
L'API implémente des validateurs Pydantic rigoureux :
- Rejet d'un champ texte vide (`""`) -> **HTTP 422 Unprocessable Entity**
- Rejet d'un champ composé d'espaces uniquement (`"   "`) -> **HTTP 422 Unprocessable Entity**
- Rejet d'un payload incomplet (champ manquant) -> **HTTP 422 Unprocessable Entity**
- Rejet d'un type de donnée corrompu (ex: nombre entier au lieu de chaîne) -> **HTTP 422 Unprocessable Entity**
- Rejet d'un paramètre `top_k` hors limites (`< 1` ou `> 10`) -> **HTTP 422 Unprocessable Entity**

### 6.3 Endpoints Disponibles

| Méthode | Route | Description | Code Succès |
|---|---|---|---|
| `GET` | `/` | Page d'accueil et liens de documentation | `200 OK` |
| `GET` | `/health` | Statut de santé, modèle chargé et périphérique | `200 OK` |
| `GET` | `/intents` | Catalogue des 77 intentions et des 7 départements | `200 OK` |
| `POST` | `/predict` | **Endpoint métier principal** : routage unitaire d'une requête | `200 OK` (ou `422`) |
| `POST` | `/predict/batch` | Routage d'une collection de requêtes en un appel | `200 OK` (ou `422`) |
| `GET` | `/docs` | Documentation interactive Swagger UI / OpenAPI | `200 OK` |

#### Exemple de Requête POST `/predict` (capture réelle, non retouchée) :
```json
{
  "text": "I lost my credit card while traveling abroad, how can I block it immediately?",
  "top_k": 3
}
```

#### Exemple de Réponse JSON (capture réelle) :
```json
{
  "query": "I lost my credit card while traveling abroad, how can I block it immediately?",
  "predicted_intent": "lost_or_stolen_card",
  "confidence": 0.165,
  "department": "Cards_Management",
  "department_description": "Gestion des cartes physiques/virtuelles, blocage, commande et code PIN",
  "top_candidates": [
    {
      "intent": "lost_or_stolen_card",
      "confidence": 0.165,
      "department": "Cards_Management"
    },
    {
      "intent": "passcode_forgotten",
      "confidence": 0.1517,
      "department": "Account_Profile"
    },
    {
      "intent": "pin_blocked",
      "confidence": 0.1332,
      "department": "Cards_Management"
    }
  ],
  "inference_time_ms": 5.83
}
```

*Confiance volontairement basse (16.5%) dans cet exemple réel : c'est un cas représentatif, pas un cas trié sur le volet — voir la section 4 pour la mesure d'accuracy globale.*

---

## 7. Client de Test & Validation HTTP (`test_api.py`)

Le script `test_api.py` utilise la bibliothèque `httpx` pour valider l'ensemble du contrat de service :
1. **Test 1 (`/health`)** : Vérification de la disponibilité du modèle et des 77 classes.
2. **Test 2 (`/intents`)** : Vérification du catalogue d'intentions et de la cohérence des 7 départements.
3. **Test 3 (`/predict` - Cas nominaux)** : Validation du routage sur 6 cas d'usage réels (Cartes, Virements, Mot de passe, Apple Pay, Distributeur ATM, Frais de change).
4. **Test 4 (`/predict/batch`)** : Validation du traitement par lot.
5. **Test 5 (Rejets stricts 422)** : Validation de 6 scénarios d'erreurs d'entrée (texte vide, espaces, types invalides, champ absent, top_k invalide).
6. **Test 6 (Benchmark de latence)** : Calcul de la latence médiane ($p_{50}$) et du 95e percentile ($p_{95}$).

**Résultat du dernier run (GPU, modèle LoRA) : 15/16 tests validés (93.8%).** L'unique échec est un routage réellement incorrect (voir section 4, Analyse des Limites) — le test le détecte parce qu'il compare le département renvoyé au département métier attendu, et non uniquement le code HTTP.

---

## 8. Guide de Lancement Rapide (Local & Docker)

### 8.1 Installation Locale & Lancement

```bash
# 1. Cloner le dépôt et installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle et exporter les artéfacts
python train.py

# 3. Démarrer l'API FastAPI en production
uvicorn app:app --host 127.0.0.1 --port 8000

# 4. Dans un second terminal, exécuter la suite de tests de preuve
python test_api.py
```

L'interface Swagger est immédiatement accessible à : **`http://127.0.0.1:8000/docs`**.

### 8.2 Déploiement Conteneurisé (Docker)

```bash
# Construction de l'image Docker
docker build -t bankroute-ai .

# Lancement du conteneur sur le port 8000
docker run -d -p 8000:8000 --name bankroute-service bankroute-ai

# Vérification du healthcheck
curl http://localhost:8000/health
```

---

## 9. Bonus Revendiqués (+1 point)

Conformément à la section 6 de l'énoncé, nous revendiquons les difficultés supplémentaires suivantes :
1. **Routage Hiérarchique Multi-Niveaux :** Classification fine sur **77 intentions** couplée à une agrégation métier vers **7 départements opérationnels**, transformant une prédiction brute en décision d'entreprise actionnable.
2. **Gestion Stricte & Documentée des Erreurs 422 :** Implémentation exhaustive de validateurs Pydantic pour intercepter les requêtes corrompues, prouvée par la suite de tests négatifs `test_api.py`.
3. **Mesure de Latence & Benchmarking :** Intégration d'un profilage de performance temps réel ($p_{50}$ et $p_{95}$) dans le service et les tests.
4. **Prêt pour la Production :** Déploiement conteneurisé avec `Dockerfile` et `HEALTHCHECK` automatisé.

---

## 10. Limites & « Avec une Semaine de Plus »

Si nous disposions d'une semaine supplémentaire, voici nos priorités d'ingénierie :
1. **Mécanisme d'Escalade Humaine (Human-in-the-loop) :** Si $\text{confiance} < 0.65$, router vers une file de révision manuelle avec suggestion des 3 meilleures hypothèses.
2. **Quantization & Accélération ONNX :** Conversion du modèle en format ONNX Runtime / INT8 pour abaisser la latence CPU sous la barre des 5 ms.
3. **Support Multilingue :** Entraînement d'un modèle basé sur `xlm-roberta-base` ou `camembert-base` pour traiter indifféremment les réclamations en français, anglais, espagnol et allemand.
4. **Observabilité & Monitoring du Data Drift :** Intégration d'un middleware Prometheus / EvidentlyAI pour suivre l'évolution des distributions d'intentions et détecter l'émergence de nouveaux motifs de réclamations.

---
*Projet réalisé pour la soutenance MLDL M2 · IPSSI 2026*
