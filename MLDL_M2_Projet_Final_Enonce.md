# Projet final · MLDL M2 : Un modèle en production

**NEEKOCODE Formation · IPSSI · MLDL M2 · MIA5 26.1 · lancé jeudi 20 août 14h00 · soutenu vendredi 21 août 15h00**
**Travail en groupe (G1 · G2 · G3) · 50 % de la note du module**

---

## 1. L'exigence

Une phrase, non négociable :

> **Votre groupe entraîne un modèle de deep learning sur un GPU qu'il provisionne, et le déploie derrière une API FastAPI. Vendredi à 14h45, l'application répond en HTTP. Sinon, la note plafonne.**

Vous avez tout construit cette semaine : l'entraînement mesuré (TP1-TP3), le transfer learning et le fine-tuning (TP2, atelier acte 1), la mesure honnête (guide métriques), le service typé (atelier acte 3), l'infrastructure GPU (guide GPU v2). Le projet assemble ces briques sur un cas que vous choisissez. Le notebook de l'atelier d'aujourd'hui est votre kit de démarrage : réutilisez-le sans scrupule.

## 2. Le menu des secteurs

Choisissez UN secteur. Les jeux de données indiqués sont des exemples vérifiés sur Hugging Face : vous pouvez en proposer un autre (validation par moi au cadrage). L'anglais des données n'est pas un obstacle : vos modèles pré-entraînés le lisent.

| Secteur | Contexte business | Données exemples | Voie modèle naturelle |
|---|---|---|---|
| **🌾 Agroalimentaire** | Diagnostic visuel de cultures : détecter une maladie sur photo de feuille avant qu'elle ne ruine la parcelle | `beans` (~1 300 photos de feuilles, 3 classes) ; sous-ensemble de `food101` (tri qualité produits) | Transfer learning CNN (ResNet, l'arc du TP2) |
| **🛰️ Environnement & territoire** | Occupation des sols par satellite : urbanisme, suivi agricole, assurance climatique | `blanchon/EuroSAT_RGB` (27 000 tuiles Sentinel-2, 10 classes) | Transfer CNN : petites images à remonter en résolution : vous connaissez le piège |
| **🏦 Banque & relation client** | Router automatiquement les demandes clients vers le bon service | `banking77` (13 000 requêtes, 77 intentions) | LoRA sur un encoder texte (l'acte 1 d'aujourd'hui) |
| **🛡️ Modération de plateforme** | Détecter les contenus problématiques : risque légal, files de modération humaine à prioriser | `tweet_eval` (config `hate` ou `offensive`) ; `go_emotions` (nuances fines) | LoRA texte ; attention au déséquilibre de classes : le guide métriques est votre ami |
| **📱 Télécom : spam & fraude** | Filtrer les messages frauduleux avant qu'ils n'atteignent l'utilisateur | `sms_spam` (5 500 SMS étiquetés) | LoRA texte, ou attention from scratch (votre MiniTransformer du TP3, adapté : voie ambitieuse, défendable) |

Interdits car déjà démontrés : sentiment Allociné, RAG sur les guides du module, CIFAR-10 brut.

## 3. Les contraintes techniques

1. **Modèle entraîné, pas prétendu.** Un entraînement réel par votre groupe, avec courbes de loss et métriques de validation dans un notebook exécuté (un notebook sans sorties n'est pas évaluable). Un appel à une API externe ne remplace jamais le modèle entraîné : il peut le compléter.
2. **GPU provisionné et documenté.** Colab T4 ou VM de groupe. Le README indique le GPU, avec preuve dans le notebook (`nvidia-smi`, `torch.cuda.is_available()`). Étiquette VM inchangée : `nvidia-smi` avant de prendre, stop en partant, budget suivi.
3. **Mesure honnête.** Split propre sans fuite, métrique adaptée au problème (classes déséquilibrées : l'accuracy seule ne suffit pas), et une baseline simple comparée (classe majoritaire ou modèle naïf) : c'est votre premier argument de soutenance.
4. **Sous-échantillonnez sans honte.** Un modèle entraîné proprement sur 5 000 exemples bat un modèle bâclé sur 500 000. Le budget temps est votre première contrainte d'architecte : dimensionnez.
5. **Service FastAPI.** Au moins un endpoint métier POST avec schémas Pydantic d'entrée et de sortie, modèle chargé une fois au lifespan, `/docs` accessible. Un script `test_api.py` (httpx) prouve que le service répond.
6. **La démo est en HTTP, en direct.** À la soutenance, le service tourne (VM de groupe recommandée, machine locale ou Colab acceptés) et répond à des appels httpx sous nos yeux : y compris une requête invalide, qui doit prendre son 422.

## 4. Les livrables (gel vendredi 14h45, dépôt Teams canal groupe)

- **Notebook d'entraînement exécuté** : données, split, entraînement, courbes, métriques, comparaison baseline. Toutes sorties visibles.
- **`app.py`** : le service. **`test_api.py`** : le client de preuve.
- **README.md** : le document qui raconte vos décisions : pourquoi ce secteur et ces données, pourquoi ce modèle plutôt que les alternatives écartées, quel GPU et pourquoi, vos mesures avec leurs limites, et ce que vous feriez avec une semaine de plus. C'est le livrable le plus discriminant.
- **L'artefact modèle** (poids ou adaptateur ; un lien si trop lourd pour Teams).
- Slides facultatives (5 maximum) : la démo vaut mieux qu'un diaporama.

## 5. Les jalons

| Quand | Quoi |
|---|---|
| **Jeudi 17h30** | **Cadrage déposé au canal groupe** : secteur, données, modèle envisagé, plan GPU, qui fait quoi. Verdict de ma part dans la soirée : GO / GO avec réserve / à retravailler. |
| Vendredi 12h00 | Démo minimale conseillée : un endpoint qui répond, même avec un modèle médiocre. Le reste de la journée améliore, il ne sauve pas. |
| **Vendredi 14h45** | **Gel et dépôt.** Plus un commit, plus une cellule. |
| Vendredi 15h00-16h30 | Soutenances : 25 minutes par groupe (grille séparée). |

## 6. Ce qui fait la note

Barème détaillé dans le document séparé. L'esprit en une ligne : **la différence entre 12 et 18, c'est la profondeur des justifications, pas le nombre de features.** Un service modeste, mesuré honnêtement et défendu avec lucidité, bat un prototype ambitieux que personne ne sait expliquer.

Bonus de difficulté : +1 point (plafonné à 20) pour une contrainte plus dure assumée et documentée : données assemblées par vous, déséquilibre de classes traité et mesuré, service exposé publiquement depuis la VM, quantization mesurée. Le README doit la revendiquer explicitement.

Bonne chance. Construisez quelque chose que vous aurez envie de défendre.

— Mor
