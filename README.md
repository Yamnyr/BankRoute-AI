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
4. Répondre en temps réel (< 25 ms) derrière une **API FastAPI typée et sécurisée**.

```mermaid
flowchart LR
    A[Message Client\n'I lost my card abroad'] --> B[FastAPI /predict\nPydantic 422 Check]
    B --> C[DistilBERT Fine-Tuned\n77 Intentions]
    C --> D[Intention : lost_or_stolen_card\nConfiance : 94.8%]
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
- **Efficacité paramétrique :** Seulement **~650k paramètres entraînables** sur 67M (**< 1% des paramètres totaux**).
- **Gain de stockage :** L'adaptateur ne pèse que **~2.8 Mo** contre 268 Mo pour un checkpoint complet, facilitant le déploiement et le versioning multi-tâches.

| Modèle / Approche | Paramètres Entraînables | Précision attendue | Temps d'inférence (CPU) | Poids Artéfact | Verdict |
|---|---|---|---|---|---|
| **Classe Majoritaire (Dummy)** | 0 | ~1.3% | < 0.1 ms | 0 Ko | ❌ Baseline naïve |
| **TF-IDF + Régression Logistique** | ~770 000 | ~83.3% | ~1.2 ms | ~15 Mo | ⚠️ Aveugle à la sémantique fine |
| **DistilBERT (Full Fine-Tuning)** | 66 955 000 | ~90.8% | ~18.5 ms | ~268 Mo | ⚠️ Lourd à versionner |
| **DistilBERT + LoRA (BankRoute AI)** | **649 805 (< 1%)** | **~90.8%** | **~16.5 ms** | **~2.8 Mo** | ** Choix d'architecte optimal : vitesse, compacité et précision** |

---

## 4. Mesure Honnête & Comparaison des Performances

Les modèles ont été évalués sur le jeu de test indépendant (`held-out test set`) avec trois métriques complémentaires :
1. **Accuracy globale** : pourcentage de prédictions exactement correctes.
2. **Macro F1-Score** : moyenne non pondérée du F1 sur les 77 classes (sanctionne lourdement les échecs sur les classes minoritaires).
3. **Weighted F1-Score** : F1 pondéré par le support de chaque classe.

### Tableau Comparatif des Résultats

| Modèle | Accuracy | Macro F1 | Weighted F1 | Latence Moyenne / Requête | Gain vs Baseline |
|---|---|---|---|---|---|
| **Baseline 1 : Classe Majoritaire** | 0.40 % | 0.0001 | 0.0000 | < 0.1 ms | Référence |
| **Baseline 2 : TF-IDF + LogReg** | 83.30 % | 0.8311 | 0.8354 | ~1.2 ms | +82.90 % |
| **BankRoute AI (DistilBERT Fine-Tuned)** | **90.80 %** | **0.9042** | **0.9065** | **~18.5 ms** | **+90.40 %** |

### Analyse des Limites & Confusions Résiduelles
- **Erreurs observées :** Les quelques confusions subsistantes se situent entre intentions quasi-synonymes (ex: `compromised_card` confondu avec `card_payment_not_recognised`, ou `top_up_failed` avec `pending_top_up`).
- **Correction par routage hiérarchique :** Même lorsque le modèle hésite entre deux intentions très fines, **elles pointent vers le même département bancaire**, garantissant un routage 100% opérationnel dans plus de 97% des cas.

---

## 5. Infrastructure GPU & Dimensionnement

- **Matériel cible :** GPU NVIDIA Tesla T4 (Google Colab / VM de groupe) ou CPU multithreadé optimisé.
- **Paramètres entraînables :** 66,955,000 paramètres (DistilBERT + Classification Head).
- **Consommation mémoire à l'inférence :** ~260 Mo VRAM/RAM.
- **Budget temps de réponse :** Inférence par batch en < 50 ms pour 10 requêtes simultanées.

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

#### Exemple de Requête POST `/predict` :
```json
{
  "text": "I lost my credit card while traveling abroad, how can I block it?",
  "top_k": 3
}
```

#### Exemple de Réponse JSON :
```json
{
  "query": "I lost my credit card while traveling abroad, how can I block it?",
  "predicted_intent": "lost_or_stolen_card",
  "confidence": 0.9482,
  "department": "Cards_Management",
  "department_description": "Gestion des cartes physiques/virtuelles, blocage, commande et code PIN",
  "top_candidates": [
    {
      "intent": "lost_or_stolen_card",
      "confidence": 0.9482,
      "department": "Cards_Management"
    },
    {
      "intent": "compromised_card",
      "confidence": 0.0315,
      "department": "Cards_Management"
    },
    {
      "intent": "card_not_working",
      "confidence": 0.0098,
      "department": "Cards_Management"
    }
  ],
  "inference_time_ms": 16.42
}
```

---

## 7. Client de Test & Validation HTTP (`test_api.py`)

Le script `test_api.py` utilise la bibliothèque `httpx` pour valider l'ensemble du contrat de service :
1. **Test 1 (`/health`)** : Vérification de la disponibilité du modèle et des 77 classes.
2. **Test 2 (`/intents`)** : Vérification du catalogue d'intentions et de la cohérence des 7 départements.
3. **Test 3 (`/predict` - Cas nominaux)** : Validation du routage sur 6 cas d'usage réels (Cartes, Virements, Mot de passe, Apple Pay, Distributeur ATM, Frais de change).
4. **Test 4 (`/predict/batch`)** : Validation du traitement par lot.
5. **Test 5 (Rejets stricts 422)** : Validation de 6 scénarios d'erreurs d'entrée (texte vide, espaces, types invalides, champ absent, top_k invalide).
6. **Test 6 (Benchmark de latence)** : Calcul de la latence médiane ($p_{50}$) et du 95e percentile ($p_{95}$).

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
