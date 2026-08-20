"""
BankRoute AI - FastAPI Production Service
API de routage intelligent des requêtes et réclamations bancaires (77 intentions, 7 départements)
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Chemins et constantes
ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "./model_artifact")
FALLBACK_MODEL = "distilbert-base-uncased"

# Variable globale pour stocker les composants chargés au lifespan
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI (Lifespan) :
    Charge le modèle, le tokenizer et les mappings en mémoire UNE SEULE FOIS au démarrage.
    Libère la mémoire à l'arrêt du service.
    """
    print(f">> [Lifespan] Initialisation du service BankRoute AI...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ml_models["device"] = device
    
    # 1. Chargement des mappings d'intentions et départements
    mapping_file = os.path.join(ARTIFACT_DIR, "intent_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            ml_models["id2label"] = {int(k): v for k, v in metadata["id2label"].items()}
            ml_models["label2id"] = metadata["label2id"]
            ml_models["department_mapping"] = metadata["department_mapping"]
            ml_models["department_descriptions"] = metadata["department_descriptions"]
            print(f">> [Lifespan] Mappings chargés : {len(ml_models['id2label'])} intentions, {len(ml_models['department_descriptions'])} départements.")
    else:
        print(f">> [Lifespan] Attention : Fichier de mapping non trouvé dans {mapping_file}. Mode fallback.")
        ml_models["id2label"] = {}
        ml_models["label2id"] = {}
        ml_models["department_mapping"] = {}
        ml_models["department_descriptions"] = {}

    # 2. Chargement du modèle et du tokenizer
    adapter_file = os.path.join(ARTIFACT_DIR, "adapter_config.json")
    config_file = os.path.join(ARTIFACT_DIR, "config.json")
    
    if os.path.exists(adapter_file):
        from peft import PeftModel, PeftConfig
        print(f">> [Lifespan] Adaptateur LoRA détecté dans {ARTIFACT_DIR}")
        peft_config = PeftConfig.from_pretrained(ARTIFACT_DIR)
        base_name = peft_config.base_model_name_or_path or FALLBACK_MODEL
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR if os.path.exists(os.path.join(ARTIFACT_DIR, "tokenizer.json")) else base_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=len(ml_models["id2label"]) if ml_models["id2label"] else 77
        )
        model = PeftModel.from_pretrained(base_model, ARTIFACT_DIR)
        ml_models["model_name"] = f"DistilBERT-Banking77-LoRA (r={peft_config.r})"
    elif os.path.exists(config_file):
        print(f">> [Lifespan] Modèle complet (Full Checkpoint) détecté dans {ARTIFACT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(ARTIFACT_DIR)
        ml_models["model_name"] = "DistilBERT-Banking77-FineTuned"
    else:
        print(f">> [Lifespan] Mode Fallback : chargement de {FALLBACK_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_MODEL,
            num_labels=len(ml_models["id2label"]) if ml_models["id2label"] else 77
        )
        ml_models["model_name"] = "DistilBERT-Base-Uncased"
    
    model.to(device)
    model.eval()
    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    
    # Préchauffage (warmup) du modèle
    dummy_input = tokenizer("Test warmup query", return_tensors="pt", max_length=64, truncation=True).to(device)
    with torch.no_grad():
        _ = model(**dummy_input)
    
    print(">> [Lifespan] Modèle préchauffé et prêt à recevoir du trafic !")
    
    yield
    
    # Nettoyage à l'arrêt
    print(">> [Lifespan] Déchargement du modèle et libération des ressources...")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialisation de l'application FastAPI avec Lifespan et documentation OpenAPI
app = FastAPI(
    title="BankRoute AI · Banking Support Routing API",
    description="API de routage intelligent des réclamations bancaires basée sur DistilBERT fine-tuné sur 77 intentions et 7 départements.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
# Note : allow_origins=["*"] est incompatible avec allow_credentials=True (rejeté par
# les navigateurs). Cette API ne pose pas de cookies/credentials, donc credentials=False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# SCHÉMAS PYDANTIC (Validation stricte des entrées et typage des sorties)
# =====================================================================

class RoutingRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Requête, réclamation ou message du client bancaire (non vide, au moins 3 caractères).",
        examples=["I lost my credit card while traveling abroad, how do I block it?"]
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de prédictions candidates alternatives à retourner (entre 1 et 10)."
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Le texte de la requête ne peut pas être composé uniquement d'espaces vides.")
        return stripped


class CandidatePrediction(BaseModel):
    intent: str = Field(..., description="Nom technique de l'intention bancaire identifiée.")
    confidence: float = Field(..., description="Probabilité estimée (Softmax entre 0.0 et 1.0).")
    department: str = Field(..., description="Département bancaire opérationnel associé.")


class RoutingResponse(BaseModel):
    query: str = Field(..., description="Texte de la requête soumise.")
    predicted_intent: str = Field(..., description="Intention principale prédite.")
    confidence: float = Field(..., description="Indice de confiance de la prédiction principale (0 à 1).")
    department: str = Field(..., description="Département bancaire vers lequel router le ticket.")
    department_description: str = Field(..., description="Rôle et périmètre du département assigné.")
    top_candidates: List[CandidatePrediction] = Field(..., description="Liste ordonnée des meilleures intentions candidates.")
    inference_time_ms: float = Field(..., description="Temps d'inférence du modèle en millisecondes.")


class BatchRoutingRequest(BaseModel):
    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Liste de requêtes à traiter en lot (1 à 50 requêtes simultanées).",
        examples=[[
            "Can you help me cancel my transfer?",
            "My contactless card payment was declined at the supermarket.",
            "I want to change my PIN code in the app."
        ]]
    )
    top_k: int = Field(default=3, ge=1, le=5, description="Nombre de candidats par requête.")

    @field_validator("queries")
    @classmethod
    def validate_queries_list(cls, queries: List[str]) -> List[str]:
        cleaned = []
        for i, q in enumerate(queries):
            if not isinstance(q, str) or not q.strip() or len(q.strip()) < 3:
                raise ValueError(f"La requête à l'index {i} est invalide (doit comporter au moins 3 caractères utiles).")
            cleaned.append(q.strip())
        return cleaned


class BatchRoutingResponse(BaseModel):
    total_queries: int
    predictions: List[RoutingResponse]
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    total_intents: int
    total_departments: int
    version: str


# =====================================================================
# FONCTIONS UTILITAIRES D'INFÉRENCE
# =====================================================================

def predict_single_query(text: str, top_k: int = 3) -> Dict:
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")
    device = ml_models.get("device", torch.device("cpu"))
    id2label = ml_models.get("id2label", {})
    dept_map = ml_models.get("department_mapping", {})
    dept_desc = ml_models.get("department_descriptions", {})

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas encore chargé en mémoire."
        )

    start_time = time.perf_counter()
    
    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding=True
    ).to(device)

    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze(0)

    # Top-K
    top_k_probs, top_k_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
    
    top_candidates = []
    for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
        intent_name = id2label.get(idx, f"intent_{idx}")
        dept = dept_map.get(intent_name, "General_Support")
        top_candidates.append({
            "intent": intent_name,
            "confidence": round(float(prob), 4),
            "department": dept
        })

    best_candidate = top_candidates[0]
    best_dept = best_candidate["department"]
    best_dept_desc = dept_desc.get(best_dept, "Support bancaire général")
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "query": text,
        "predicted_intent": best_candidate["intent"],
        "confidence": best_candidate["confidence"],
        "department": best_dept,
        "department_description": best_dept_desc,
        "top_candidates": top_candidates,
        "inference_time_ms": round(elapsed_ms, 2)
    }


def predict_batch_queries(texts: List[str], top_k: int = 3) -> List[Dict]:
    """Inférence par lot : un seul forward pass tenseur pour toutes les requêtes,
    au lieu d'une boucle d'inférences unitaires."""
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")
    device = ml_models.get("device", torch.device("cpu"))
    id2label = ml_models.get("id2label", {})
    dept_map = ml_models.get("department_mapping", {})
    dept_desc = ml_models.get("department_descriptions", {})

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas encore chargé en mémoire."
        )

    start_time = time.perf_counter()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    per_query_ms = round(elapsed_ms / len(texts), 2)

    results = []
    for text, probs in zip(texts, probabilities):
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        top_candidates = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_indices.tolist()):
            intent_name = id2label.get(idx, f"intent_{idx}")
            dept = dept_map.get(intent_name, "General_Support")
            top_candidates.append({
                "intent": intent_name,
                "confidence": round(float(prob), 4),
                "department": dept
            })

        best_candidate = top_candidates[0]
        best_dept = best_candidate["department"]
        best_dept_desc = dept_desc.get(best_dept, "Support bancaire général")

        results.append({
            "query": text,
            "predicted_intent": best_candidate["intent"],
            "confidence": best_candidate["confidence"],
            "department": best_dept,
            "department_description": best_dept_desc,
            "top_candidates": top_candidates,
            "inference_time_ms": per_query_ms
        })

    return results


# =====================================================================
# ENDPOINTS DE L'API FASTAPI
# =====================================================================

@app.get("/", summary="Page d'accueil de l'API")
def root():
    return {
        "message": "Bienvenue sur l'API BankRoute AI - Routage intelligent du support bancaire.",
        "documentation": "/docs",
        "health_check": "/health",
        "intents_catalog": "/intents"
    }


@app.get("/health", response_model=HealthResponse, summary="Vérification de l'état de santé du service")
def health_check():
    return HealthResponse(
        status="healthy" if "model" in ml_models else "uninitialized",
        model_name=ml_models.get("model_name", "None"),
        device=str(ml_models.get("device", "cpu")),
        total_intents=len(ml_models.get("id2label", {})),
        total_departments=len(ml_models.get("department_descriptions", {})),
        version="1.0.0"
    )


@app.get("/intents", summary="Catalogue des intentions et départements bancaires")
def get_intents_catalog():
    return {
        "total_intents": len(ml_models.get("id2label", {})),
        "total_departments": len(ml_models.get("department_descriptions", {})),
        "departments": ml_models.get("department_descriptions", {}),
        "intents": ml_models.get("id2label", {}),
        "intent_to_department": ml_models.get("department_mapping", {})
    }


@app.post(
    "/predict",
    response_model=RoutingResponse,
    status_code=status.HTTP_200_OK,
    summary="Router une requête client vers une intention et un département",
    responses={
        200: {"description": "Prédiction et routage calculés avec succès."},
        422: {"description": "Validation Error : requête invalide (texte vide, types incorrects, etc.)."}
    }
)
def predict_intent(request: RoutingRequest):
    """
    Point de terminaison principal :
    - Reçoit le texte de la requête bancaire
    - Détermine l'intention métier parmi les 77 classes banking77
    - Associe automatiquement le département bancaire opérationnel compétent
    - Calcule l'indice de confiance et les alternatives candidates
    """
    return predict_single_query(request.text, top_k=request.top_k)


@app.post(
    "/predict/batch",
    response_model=BatchRoutingResponse,
    status_code=status.HTTP_200_OK,
    summary="Router un lot de requêtes bancaires",
    responses={
        200: {"description": "Lot de requêtes traité avec succès."},
        422: {"description": "Validation Error : liste vide ou requête corrompue."}
    }
)
def predict_batch(request: BatchRoutingRequest):
    """
    Point de terminaison par lot :
    Traite une collection de requêtes en une seule requête HTTP.
    """
    start_total = time.perf_counter()
    results = predict_batch_queries(request.queries, top_k=request.top_k)
    total_ms = (time.perf_counter() - start_total) * 1000.0
    
    return BatchRoutingResponse(
        total_queries=len(results),
        predictions=results,
        total_processing_time_ms=round(total_ms, 2)
    )
