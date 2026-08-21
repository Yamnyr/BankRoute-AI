# 📊 Diaporama de Soutenance · BankRoute AI (5 Slides)
**Projet Final MLDL M2 · IPSSI (MIA5 26.1)**  
**Équipe :** Marine · Corentin · Quentin  
**Format :** 5 slides structurées par rôle & thématique

---

## 🖥️ Slide 1 : Contexte — Problématique Métier, Équipe & Flow de Routage

### 🎯 La Problématique Métier
- **Le constat :** Des dizaines de milliers de sollicitations quotidiennes par chat et email dans le secteur bancaire et fintech.
- **La friction :** Un mauvais aiguillage génère de l'attente, de la frustration client et un surcoût de transfert entre services.
- **La solution BankRoute AI :** Un micro-service IA capable de classifier instantanément une demande parmi **77 intentions fines** et de la router vers le **bon département métier** en **< 6 ms**.

### 👥 L'Équipe & Répartition des Rôles
- **Corentin :** Data Pipeline, baselines, modélisation LoRA, notebook d'entraînement & métriques.
- **Quentin :** Architecture logicielle FastAPI, gestionnaire de lifespan, validation stricte 422 & batching.
- **Marine :** Suite de tests automatisée `httpx`, benchmark de latence, Dockerfile & documentation.

### 🔄 Le Flow Visuel de Routage
```
[Message Client] "I lost my card abroad"
       │
       ▼
[FastAPI /predict] ── (Validation Pydantic / Rejet 422 si invalide)
       │
       ▼
[DistilBERT + LoRA (PEFT)] ── (Inférence temps réel 5.9 ms)
       │
       ▼
[Intention Prédite] "lost_or_stolen_card" (Confiance : 94.8%)
       │
       ▼
[Routage Opérationnel] "Cards_Management" (Service Cartes & Sécurité)
```

---

## 🖥️ Slide 2 : Corentin — Données, Baselines, LoRA & Performances
### **Pipeline Data & Modélisation Transformer**

### 📦 Le Dataset `banking77` & Split Propre
- **13 082 requêtes clients réelles** étiquetées en anglais sur **77 classes fines**.
- **Split sans fuite (Zero Leakage) :** Train (90%) / Val (10%) internes, et évaluation sur un Test set indépendant de **3 076 exemples** (`held-out`).

### ⚡ L'Arbitrage LoRA (Low-Rank Adaptation - PEFT)
- **Pourquoi LoRA ?** Gel des 67M de poids du Transformer et injection de matrices de bas rang ($r=16, \alpha=32$) sur `q_lin` et `v_lin`.
- **Efficacité paramétrique :** **944 717 paramètres entraînés (~1.4%)** $\rightarrow$ adaptateur léger de **3.78 Mo** (68× plus compact que les 256 Mo du full checkpoint).

### 📈 Résultats & Métriques
- **Baseline 1 (Classe Majoritaire) :** 1.30 % accuracy / 0.0003 Macro F1
- **Baseline 2 (TF-IDF + Logistic Regression) :** 88.52 % accuracy / 0.8856 Macro F1
- **BankRoute AI (DistilBERT + LoRA) :**
  - 🎯 **91.03 % d'Accuracy au niveau Intention** (77 classes, Macro F1 : 0.9103)
  - 🏢 **95.51 % d'Accuracy au niveau Département** (sécurité opérationnelle renforcée)

---

## 🖥️ Slide 3 : Quentin — Endpoints FastAPI, Validation Pydantic & Exemple Réel
### **Architecture Logicielle & Service de Production (`app.py`)**

### 🛡️ Les Fondations du Service
1. **Lifespan (`@asynccontextmanager`) :** Modèle, tokenizer et mappings chargés **une seule fois au démarrage** en mémoire vive + préchauffage (*warmup*). Zéro latence au 1er appel client.
2. **Validation Stricte Pydantic (HTTP 422) :** Rejet immédiat des requêtes vides `""`, espaces seuls `"   "`, mauvais types (`12345`) ou `top_k` hors bornes $[1, 10]$.
3. **Inférence Batch Vectorisée (`/predict/batch`) :** Un seul forward pass tensoriel PyTorch parallélisé pour tout le lot.

### 📋 Catalogue des Endpoints
- `GET /health` : Statut de santé et informations du modèle chargé.
- `GET /intents` : Catalogue des 77 intentions et des 7 départements.
- `POST /predict` & `POST /predict/batch` : Routage unitaire et par lot.
- `GET /docs` : Documentation Swagger UI interactive.

### 📄 Exemple Réel de Réponse JSON (`POST /predict`)
```json
{
  "query": "I lost my credit card while traveling abroad, how can I block it immediately?",
  "predicted_intent": "lost_or_stolen_card",
  "confidence": 0.9482,
  "department": "Cards_Management",
  "department_description": "Gestion des cartes physiques/virtuelles, blocage, commande et code PIN",
  "top_candidates": [
    {"intent": "lost_or_stolen_card", "confidence": 0.9482, "department": "Cards_Management"},
    {"intent": "compromised_card", "confidence": 0.0315, "department": "Cards_Management"},
    {"intent": "card_not_working", "confidence": 0.0098, "department": "Cards_Management"}
  ],
  "inference_time_ms": 5.88
}
```

---

## 🖥️ Slide 4 : Marine — Validation HTTP, Latence & Docker
### **Preuve de Service, Performance & Conteneurisation**

### 🧪 Suite de Tests Automatisée `test_api.py` (httpx)
- **15 / 16 tests validés avec succès (93.8%)** en conditions réelles.
- **Validation fonctionnelle :** Health check, catalogue, routage nominal et batch.
- **Validation négative :** 6 cas de rejets stricts en **HTTP 422 Unprocessable Entity** validés.
- **Mesure honnête de l'échec :** Le test détecte une vraie confusion sémantique entre frais de change et de carte (*"unexpected extra fee charged for currency exchange"* routé vers `Cards_Management`).

### ⚡ Benchmark de Latence d'Inférence
- **Latence Médiane ($p_{50}$) :** **~5.9 ms** par requête sur GPU (et ~15.6 ms sur CPU).
- **Latence 95e percentile ($p_{95}$) :** **~6.8 ms**.
- **Traitement Batch :** 3 requêtes traitées en **25 ms au global** (~8.3 ms/requête).

### 🐳 Déploiement Conteneurisé (Docker)
- **`Dockerfile` optimisé :** Utilisation des wheels PyTorch CPU (`--extra-index-url`) réduisant la taille de l'image de 9.5 Go à **2.9 Go**.
- **Healthcheck & Port dynamique :** Support automatique des variables d'environnement (`${PORT:-8000}`).

---

## 🖥️ Slide 5 : Bilan — Bonus Revendiqués, Limites & Démo en Direct
### **Synthèse Finale & Perspectives**

### ⭐ Bonus Revendiqués (+1 pt)
1. **Routage Hiérarchique Métier :** 77 intentions $\rightarrow$ 7 départements opérationnels (atteignant **95.51 %** d'accuracy départementale).
2. **Gestion Stricte & Prouvée des Erreurs 422 :** Couverture complète des cas limites dans `test_api.py`.
3. **Latence Profilée & Conteneurisation :** Mesure $p_{50}/p_{95}$ en conditions réelles et Dockerfile de production.

### 🔍 Limites Reconnues & « Avec Une Semaine de Plus »
- **Human-in-the-loop :** Escalade vers un conseiller humain si $\text{confiance} < 0.65$.
- **Quantization ONNX INT8 :** Inférence CPU sous la barre des 3 ms.
- **Support Multilingue :** Entraînement avec CamemBERT / XLM-RoBERTa pour gérer les réclamations en français et espagnol.

---

### 🎬 Place à la Démonstration en Direct !
- 🌐 **Documentation Swagger :** `http://127.0.0.1:8000/docs`
- ⚡ **Suite de preuve HTTP :** `python test_api.py`
