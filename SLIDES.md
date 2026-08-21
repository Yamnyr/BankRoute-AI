# 📊 Diaporama de Soutenance · BankRoute AI (5 Slides)
**Projet Final MLDL M2 · IPSSI (MIA5 26.1)**  
**Équipe :** Marine · Corentin · Quentin  
**Format :** 5 slides maximum (Conforme aux exigences de l'énoncé)

---

## 🖥️ Slide 1 : Titre & Contexte Métier
### **BankRoute AI · Routage Intelligent du Support Bancaire**
*Classification fine sur 77 intentions & routage opérationnel en temps réel*

#### 🎯 L'Enjeu Business
- **Le constat :** Des dizaines de milliers de sollicitations quotidiennes par chat et email dans les banques et fintechs.
- **La friction :** Un mauvais aiguillage (ex: envoyer une contestation de fraude au service des relevés) génère de la latence, de la frustration client et un surcoût opérationnel majeur.
- **La solution BankRoute AI :** Un micro-service IA capable de classifier instantanément une demande parmi **77 intentions précises** et de la router vers le **bon département métier** en **< 6 ms**.

#### 👥 Répartition des Rôles
- **Corentin :** Data Pipeline, baselines, fine-tuning LoRA, métriques & notebook d'évaluation.
- **Quentin :** Architecture logicielle FastAPI, gestionnaire de lifespan, validation stricte 422 & batching.
- **Marine :** Suite de tests automatisée `httpx`, benchmark de latence, Docker & documentation.

> 🗣️ **Temps de parole recommandé :** 2 minutes (Introduction du problème et de l'équipe).

---

## 🖥️ Slide 2 : Données & Innovation Métier (Routage Hiérarchique)
### **Du Dataset Technique au Routage Opérationnel**

#### 📦 Le Jeu de Données `banking77`
- **13 082 requêtes clients réelles** étiquetées en anglais.
- **77 classes fines** : frontières sémantiques très proches (ex: `lost_or_stolen_card` vs `compromised_card`).
- **Split sans fuite :** Train (90%) / Validation (10%) internes, et évaluation sur un Test set indépendant de **3 076 exemples** (`held-out`).

#### 🏢 Notre Surcouche Métier : Les 7 Départements
Nous agrégeons les 77 intentions en **7 départements opérationnels** :
1. 💳 **`Cards_Management`** : Cartes physiques/virtuelles, blocage, commande, PIN (22 intentions).
2. 💸 **`Payments_Transfers`** : Virements nationaux/internationaux, prélèvements (11 intentions).
3. 👤 **`Account_Profile`** : Profil client, vérification d'identité KYC, clôture (13 intentions).
4. 🏧 **`Cash_ATM`** : Retraits distributeurs, pannes d'ATM, espèces (7 intentions).
5. 📥 **`Top_Up_Deposits`** : Recharges de compte, dépôts d'espèces/chèques (10 intentions).
6. 💱 **`Currency_Exchange`** : Opérations multidevises, taux de change, litiges (6 intentions).
7. 📱 **`Digital_Services`** : Apple Pay, Google Pay, services connectés (8 intentions).

> 🗣️ **Temps de parole recommandé :** 3 minutes (Explication de la valeur ajoutée métier).

---

## 🖥️ Slide 3 : Architecture Technique & Le Choix de LoRA (PEFT)
### **Pourquoi DistilBERT + LoRA ? (L'Arbitrage d'Ingénierie)**

#### ⚖️ Comparaison des Alternatives
- **Pourquoi pas un LLM (GPT-4 / Llama 3) ?** Latence prohibitive (> 500 ms vs 5.9 ms), coût au token récurrent, risque d'hallucination, confidentialité bancaire (RGPD).
- **Pourquoi pas BERT-large ?** Trop lourd (~1.3 Go) et trop lent pour du support temps réel.
- **Pourquoi DistilBERT ?** Version allégée par distillation : **60% plus rapide**, **40% plus léger**, conserve **97% des capacités linguistiques**.

#### ⚡ La Puissance de LoRA (Low-Rank Adaptation)
- **Principe :** On gèle les poids du Transformer ($W_0$) et on injecte des matrices de bas rang ($A, B$) avec **$r=16, \alpha=32$** sur les projections d'attention `q_lin` et `v_lin`.
- **L'argument choc (Chiffres réels) :**
  - **Full Fine-Tuning :** 67 012 685 paramètres (100%) $\rightarrow$ Checkpoint de **256 Mo** $\rightarrow$ Accuracy : **91.84%**
  - **DistilBERT + LoRA :** **944 717 paramètres (~1.4%)** $\rightarrow$ Adaptateur de **3.78 Mo** $\rightarrow$ Accuracy : **91.03%**
- **Conclusion :** Un artéfact **68× plus léger** pour seulement 0.81 pt d'écart. En production, LoRA permet d'héberger des dizaines d'adaptateurs spécialisés sur un même serveur.

> 🗣️ **Temps de parole recommandé :** 4 minutes (Cœur technique et démonstration de maîtrise).

---

## 🖥️ Slide 4 : Mesure Honnête des Performances
### **Résultats Comparatifs & Analyse des Limites**

#### 📈 Tableau Comparatif (Test Set Indépendant)
| Modèle / Approche | Accuracy | Macro F1 | Weighted F1 | Poids Artéfact | Latence Moyenne |
|---|---|---|---|---|---|
| **1. Classe Majoritaire (Dummy)** | 1.30 % | 0.0003 | 0.0003 | 0 Ko | < 0.1 ms |
| **2. TF-IDF + Logistic Regression** | 88.52 % | 0.8856 | 0.8857 | ~15 Mo | ~1.2 ms |
| **3. BankRoute AI (DistilBERT + LoRA)** | **91.03 %** | **0.9103** | **0.9103** | **3.78 Mo** | **~5.9 ms (GPU)** |

#### 🔍 Analyse Honnête des Limites & Confusions
- **Pourquoi le Macro F1 ?** Traite les 77 classes équitablement : un score de **0.9103** prouve l'absence d'angle mort sur les classes rares.
- **Exemple réel de confusion analysé :** La requête *"Why is there an unexpected extra fee charged on my statement for currency exchange?"* a été routée vers `Cards_Management` (`card_payment_wrong_exchange_rate`) au lieu de `Currency_Exchange`.
- **Enseignement :** La confusion sémantique entre frais de carte et frais de change traverse parfois la frontière départementale. En production, un seuil de confiance avec escalade humaine résout ce cas.

> 🗣️ **Temps de parole recommandé :** 3 minutes (Démonstration d'honnêteté scientifique).

---

## 🖥️ Slide 5 : Service FastAPI de Production & Démo
### **Architecture Logicielle, Déploiement & Démo en Direct**

#### 🛡️ Les 3 Piliers du Service `app.py`
1. **Lifespan (`@asynccontextmanager`)** : Modèle chargé une seule fois au démarrage + préchauffage (*warmup*). Zéro latence d'initialisation pour les clients.
2. **Validation Stricte Pydantic (Code HTTP 422)** : Rejet immédiat des requêtes vides, espaces seuls, mauvais types ou paramètres `top_k` invalides.
3. **Inférence Batch Vectorisée (`/predict/batch`)** : Un seul forward pass tensoriel PyTorch pour tout le lot (3 requêtes traitées en 25 ms au global).

#### 🚀 Perspectives (« Avec une Semaine de Plus »)
- **Human-in-the-loop :** File de révision manuelle si $\text{confiance} < 0.65$.
- **Quantization ONNX INT8 :** Inférence CPU sous la barre des 3 ms.
- **Multilingue :** CamemBERT / XLM-RoBERTa pour gérer les réclamations en français et espagnol.

---

### 🎬 Place à la Démonstration en Direct !
1. 🌐 Consultation de l'API interactive : **`http://127.0.0.1:8000/docs`**
2. ⚡ Exécution de la suite de preuve : **`python test_api.py`** (Cas nominaux, Batch, Rejets 422, Latence).
