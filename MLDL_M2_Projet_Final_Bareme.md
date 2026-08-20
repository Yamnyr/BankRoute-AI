# Projet final · MLDL M2 : Barème

**MIA5 26.1 · projet = 50 % de la note du module · note sur 20 · note de groupe**

---

## Les 4 axes

### 🔧 Fonctionnel · 8 points
| Critère | Points | Ce qui est regardé |
|---|---|---|
| Le service répond en direct | 3 | Endpoint métier qui répond en HTTP à la soutenance ; requête invalide rejetée en 422 ; `/docs` accessible ; modèle chargé au lifespan |
| Le modèle est entraîné, pas prétendu | 3 | Entraînement réel sur GPU documenté : courbes de loss, métriques de validation, preuve GPU dans le notebook exécuté |
| Pipeline de données propre | 2 | Split sans fuite, prétraitement justifié, dimensionnement assumé (sous-échantillon défendu) |

### 📖 Qualité code & narration · 4 points
| Critère | Points | Ce qui est regardé |
|---|---|---|
| Le README raconte les décisions | 2 | Pourquoi ces données, ce modèle, ce GPU ; alternatives écartées et pourquoi ; limites nommées ; « avec une semaine de plus » |
| Code lisible et structuré | 2 | `app.py` propre, `test_api.py` fourni, notebook ordonné, commentaires utiles |

### 🧠 Compréhension architecturale · 4 points
| Critère | Points | Ce qui est regardé |
|---|---|---|
| Trade-offs défendus | 2 | Chaque choix tient face à « pourquoi pas X ? » : modèle vs alternatives, voie choisie (transfer / LoRA / attention), coût d'inférence |
| Mesure honnête | 2 | Métrique adaptée au problème, baseline comparée, écarts expliqués sans maquillage, variance reconnue |

### 🎤 Soutenance · 4 points
| Critère | Points | Ce qui est regardé |
|---|---|---|
| Démo maîtrisée | 2 | L'URL d'abord, un scénario préparé, la panne gérée avec calme si elle arrive |
| Réponses aux questions | 2 | Chaque membre répond à au moins une question ciblée sur sa partie ; précision et honnêteté (« je ne sais pas, voici comment je vérifierais » vaut mieux qu'un bluff) |

## Plafonds (appliqués après le total)

| Situation | Note maximale |
|---|---|
| L'application ne répond pas en HTTP à la soutenance | **10 / 20** |
| Le modèle n'a pas été entraîné par le groupe (API seule, poids téléchargés sans entraînement, zero-shot) | **8 / 20** |
| Notebook d'entraînement déposé sans sorties | critère « modèle entraîné » = 0 (non évaluable), règle inchangée depuis le TP1 |
| Dépôt après 14h45 | -1 point par tranche de 15 minutes entamée |

## Bonus

**+1 point** (total plafonné à 20) pour une difficulté supplémentaire assumée, revendiquée dans le README et vérifiable : données assemblées par le groupe, déséquilibre de classes traité et mesuré (F1, matrice de confusion), service exposé publiquement depuis la VM de groupe, quantization ou latence mesurée.

## Politique de note

- **Note de groupe, sans différenciation individuelle par défaut.**
- Ajustement individuel possible **uniquement** sur éléments objectivés en soutenance : incapacité manifeste et répétée à expliquer sa propre partie lors des questions nominatives. L'ajustement est un plafonnement individuel motivé, jamais une prime.
- Les zones grises de l'énoncé se résolvent en faveur des étudiants, comme toujours dans ce module.
