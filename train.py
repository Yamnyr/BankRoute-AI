"""
BankRoute AI - Training & Evaluation Pipeline
Projet Final MLDL M2 - IPSSI

Pipeline complet :
1. Diagnostic matériel (GPU / CUDA / CPU)
2. Chargement des données banking77 & mapping hiérarchique en 7 départements bancaires
3. Split propre sans fuite de données (Train / Val / Test)
4. Entraînement de la baseline (Dummy Classifier + TF-IDF Logistic Regression)
5. Fine-tuning du modèle Transformer (DistilBERT Sequence Classification)
6. Évaluation comparative honnête (Accuracy, Macro F1, Weighted F1, matrice de confusion)
7. Export des artefacts de production dans ./model_artifact
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

# Fixer la graine pour reproductibilité

set_seed(42)

# 1) Configuration des départements bancaires

DEPARTMENT_MAPPING = {

    # 1. Cartes & Sécurité physique
    "activate_my_card": "Cards_Management",
    "card_about_to_expire": "Cards_Management",
    "card_acceptance": "Cards_Management",
    "card_arrival": "Cards_Management",
    "card_delivery_estimate": "Cards_Management",
    "card_linking": "Cards_Management",
    "card_not_working": "Cards_Management",
    "card_payment_fee_charged": "Cards_Management",
    "card_payment_not_recognised": "Cards_Management",
    "card_payment_wrong_exchange_rate": "Cards_Management",
    "card_swallowed": "Cards_Management",
    "compromised_card": "Cards_Management",
    "contactless_not_working": "Cards_Management",
    "declined_card_payment": "Cards_Management",
    "getting_spare_card": "Cards_Management",
    "getting_virtual_card": "Cards_Management",
    "lost_or_stolen_card": "Cards_Management",
    "order_physical_card": "Cards_Management",
    "pending_card_payment": "Cards_Management",
    "pin_blocked": "Cards_Management",
    "reverted_card_payment?": "Cards_Management",
    "virtual_card_not_working": "Cards_Management",


    # 2. Paiements & Virements
    "beneficiary_not_allowed": "Payments_Transfers",
    "cancel_transfer": "Payments_Transfers",
    "declined_transfer": "Payments_Transfers",
    "direct_debit_payment_not_recognised": "Payments_Transfers",
    "failed_transfer": "Payments_Transfers",
    "pending_transfer": "Payments_Transfers",
    "receiving_money": "Payments_Transfers",
    "transfer_fee_charged": "Payments_Transfers",
    "transfer_into_account": "Payments_Transfers",
    "transfer_not_received_by_recipient": "Payments_Transfers",
    "transfer_timing": "Payments_Transfers",


    # 3. Compte & Profil client
    "age_limit": "Account_Profile",
    "change_pin": "Account_Profile",
    "country_support": "Account_Profile",
    "edit_personal_details": "Account_Profile",
    "get_disposable_virtual_card": "Account_Profile",
    "get_physical_card": "Account_Profile",
    "passcode_forgotten": "Account_Profile",
    "terminate_account": "Account_Profile",
    "verify_my_identity": "Account_Profile",
    "verify_source_of_funds": "Account_Profile",
    "verify_top_up": "Account_Profile",
    "why_verify_identity": "Account_Profile",
    "unable_to_verify_identity": "Account_Profile",


    # 4. Retraits & Distributeurs ATM
    "atm_support": "Cash_ATM",
    "cash_withdrawal_charge": "Cash_ATM",
    "cash_withdrawal_not_recognised": "Cash_ATM",
    "declined_cash_withdrawal": "Cash_ATM",
    "pending_cash_withdrawal": "Cash_ATM",
    "wrong_amount_of_cash_received": "Cash_ATM",
    "wrong_exchange_rate_for_cash_withdrawal": "Cash_ATM",


    # 5. Recharges & Dépôts
    "automatic_top_up": "Top_Up_Deposits",
    "balance_not_updated_after_bank_transfer": "Top_Up_Deposits",
    "balance_not_updated_after_cheque_or_cash_deposit": "Top_Up_Deposits",
    "pending_top_up": "Top_Up_Deposits",
    "top_up_by_bank_transfer_charge": "Top_Up_Deposits",
    "top_up_by_card_charge": "Top_Up_Deposits",
    "top_up_by_cash_or_cheque": "Top_Up_Deposits",
    "top_up_failed": "Top_Up_Deposits",
    "top_up_limits": "Top_Up_Deposits",
    "top_up_reverted": "Top_Up_Deposits",
    "topping_up_by_card": "Top_Up_Deposits",


    # 6. Devises & Taux de change / Remboursements
    "exchange_charge": "Currency_Exchange",
    "exchange_rate": "Currency_Exchange",
    "exchange_via_app": "Currency_Exchange",
    "extra_charge_on_statement": "Currency_Exchange",
    "Refund_not_showing_up": "Currency_Exchange",
    "request_refund": "Currency_Exchange",
    "transaction_charged_twice": "Currency_Exchange",


    # 7. Services Digitaux & Compatibilité
    "apple_pay_or_google_pay": "Digital_Services",
    "disposable_card_limits": "Digital_Services",
    "supported_cards_and_currencies": "Digital_Services",
    "fiat_currency_support": "Digital_Services",
    "lost_or_stolen_phone": "Digital_Services",
    "visa_or_mastercard": "Digital_Services"
}

DEPARTMENT_DESCRIPTIONS = {
    "Cards_Management": "Gestion des cartes physiques/virtuelles, blocage, commande et code PIN",
    "Payments_Transfers": "Virements bancaires nationaux et internationaux, prélèvements et bénéficiaires",
    "Account_Profile": "Informations personnelles, vérification d'identité (KYC) et clôture",
    "Cash_ATM": "Retraits d'espèces, distributeurs automatiques et frais associés",
    "Top_Up_Deposits": "Alimentation du compte, dépôts d'espèces/chèques et recharges automatiques",
    "Currency_Exchange": "Opérations multidevises, taux de change et litiges de facturation/remboursement",
    "Digital_Services": "Paiements sans contact mobiles (Apple/Google Pay) et services connectés"
}


def check_environment():
    print("=" * 60)
    print("1. DIAGNOSTIC MATÉRIEL & ENVIRONNEMENT")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA disponible : {cuda_available}")
    print(f"Nombre de GPU   : {device_count}")
    print(f"Périphérique    : {device_name}")
    device = torch.device("cuda" if cuda_available else "cpu")
    return device


def load_and_prepare_data():
    print("\n" + "=" * 60)
    print("2. CHARGEMENT & PRÉPARATION DES DONNÉES (banking77)")
    print("=" * 60)
    raw_dataset = load_dataset("mteb/banking77")
    
    # Récupérer les labels uniques ordonnés
    train_texts = list(raw_dataset["train"]["text"])
    train_labels = list(raw_dataset["train"]["label"])
    train_label_texts = list(raw_dataset["train"]["label_text"])
    
    test_texts = list(raw_dataset["test"]["text"])
    test_labels = list(raw_dataset["test"]["label"])
    test_label_texts = list(raw_dataset["test"]["label_text"])
    

    raw_id_to_text = dict(zip(train_labels + test_labels, train_label_texts + test_label_texts))
    unique_intents = [raw_id_to_text[i] for i in sorted(raw_id_to_text)]
    id2label = {i: label for i, label in raw_id_to_text.items()}
    label2id = {label: i for i, label in raw_id_to_text.items()}
    
    print(f"Total exemples d'entraînement : {len(train_texts)}")
    print(f"Total exemples de test         : {len(test_texts)}")
    print(f"Nombre d'intentions (classes)  : {len(unique_intents)}")
    print(f"Nombre de départements         : {len(DEPARTMENT_DESCRIPTIONS)}")

    
    # Split Train / Validation propre sans fuite (90% train, 10% val sur le split d'entraînement)
    train_val_split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    test_data = raw_dataset["test"]
    
  
    if not torch.cuda.is_available():
        print(">> Mode CPU détecté : sous-échantillonnage stratifié assumé (3 500 exemples train, 500 val, 1 000 test) pour garantir un entraînement rapide et reproductible.")
        train_data = train_data.shuffle(seed=42).select(range(min(3500, len(train_data))))
        val_data = val_data.shuffle(seed=42).select(range(min(500, len(val_data))))
        test_data = test_data.shuffle(seed=42).select(range(min(1000, len(test_data))))
    
    print(f"\nRépartition des splits :")
    print(f"  - Train : {len(train_data)} exemples")
    print(f"  - Val   : {len(val_data)} exemples")
    print(f"  - Test  : {len(test_data)} exemples")
    
    return train_data, val_data, test_data, id2label, label2id, unique_intents


def train_baselines(train_data, val_data, test_data):
    print("\n" + "=" * 60)
    print("3. ENTRAÎNEMENT & ÉVALUATION DES BASELINES")
    print("=" * 60)
    
    X_train = train_data["text"]
    y_train = train_data["label"]
    X_test = test_data["text"]
    y_test = test_data["label"]

    
    # Baseline 1 : Dummy Classifier 
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_acc = accuracy_score(y_test, y_pred_dummy)
    dummy_f1_macro = f1_score(y_test, y_pred_dummy, average="macro", zero_division=0)
    dummy_f1_weighted = f1_score(y_test, y_pred_dummy, average="weighted", zero_division=0)
    
    print(f"[Baseline 1 - Classe Majoritaire]")
    print(f"  - Accuracy     : {dummy_acc * 100:.2f}%")
    print(f"  - Macro F1     : {dummy_f1_macro:.4f}")
    print(f"  - Weighted F1  : {dummy_f1_weighted:.4f}")

    
    # Baseline 2 : TF-IDF + Régression Logistique
    print(f"\n[Baseline 2 - TF-IDF (unigram+bigram) + Logistic Regression]")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
    start_time = time.time()
    clf.fit(X_train_vec, y_train)
    train_duration = time.time() - start_time
    
    y_pred_lr = clf.predict(X_test_vec)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1_macro = f1_score(y_test, y_pred_lr, average="macro", zero_division=0)
    lr_f1_weighted = f1_score(y_test, y_pred_lr, average="weighted", zero_division=0)
    
    print(f"  - Temps d'entraînement : {train_duration:.2f} s")
    print(f"  - Accuracy             : {lr_acc * 100:.2f}%")
    print(f"  - Macro F1             : {lr_f1_macro:.4f}")
    print(f"  - Weighted F1          : {lr_f1_weighted:.4f}")
    
    baselines_results = {
        "dummy": {
            "accuracy": float(dummy_acc),
            "macro_f1": float(dummy_f1_macro),
            "weighted_f1": float(dummy_f1_weighted)
        },
        "tfidf_logreg": {
            "accuracy": float(lr_acc),
            "macro_f1": float(lr_f1_macro),
            "weighted_f1": float(lr_f1_weighted),
            "train_duration_sec": float(train_duration)
        }
    }
    return baselines_results


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


def train_deep_learning_model(train_data, val_data, test_data, id2label, label2id, model_name="distilbert-base-uncased", num_epochs=3, use_lora=True):
    print("\n" + "=" * 60)
    mode_str = "PEFT / LoRA (Low-Rank Adaptation)" if use_lora else "Full Fine-Tuning"
    print(f"4. ENTRAÎNEMENT DU MODÈLE TRANSFORMER ({model_name} · {mode_str})")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64)
    
    tokenized_train = train_data.map(tokenize_fn, batched=True)
    tokenized_val = val_data.map(tokenize_fn, batched=True)
    tokenized_test = test_data.map(tokenize_fn, batched=True)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label={str(k): v for k, v in id2label.items()},
        label2id={v: k for k, v in id2label.items()}
    )
    
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        print(">> Configuration de l'adaptateur LoRA (PEFT) :")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            bias="none",
            modules_to_save=["classifier", "pre_classifier"]
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
    else:
        model = base_model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres totaux : {total_params:,} | Paramètres entraînables : {trainable_params:,}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    output_dir = "./training_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32 if torch.cuda.is_available() else 16,
        per_device_eval_batch_size=64 if torch.cuda.is_available() else 32,
        learning_rate=2e-4 if use_lora else 5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Lancement du fine-tuning...")
    start_train_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_train_time
    print(f"Entraînement terminé en {total_train_time:.2f} s ({total_train_time/60:.2f} min)")

    
    # Évaluation complète sur le jeu de test
    print("\n" + "=" * 60)
    print("5. ÉVALUATION FINALE SUR LE JEU DE TEST")
    print("=" * 60)
    
    start_eval_time = time.time()
    test_predictions = trainer.predict(tokenized_test)
    eval_duration = time.time() - start_eval_time
    
    test_logits = test_predictions.predictions
    test_preds = np.argmax(test_logits, axis=1)
    test_labels = test_predictions.label_ids
    
    dl_acc = accuracy_score(test_labels, test_preds)
    dl_f1_macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    dl_f1_weighted = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    
    print(f"[Modèle Deep Learning - {model_name}]")
    print(f"  - Test Accuracy : {dl_acc * 100:.2f}%")
    print(f"  - Test Macro F1 : {dl_f1_macro:.4f}")
    print(f"  - Test Weight F1: {dl_f1_weighted:.4f}")
    print(f"  - Latence moyenne par requête test : {(eval_duration / len(test_labels)) * 1000:.2f} ms")

    # Matrice de confusion au niveau des 77 intentions (granularité fine)
    cm_intents = confusion_matrix(test_labels, test_preds, labels=list(range(len(id2label))))

    # Matrice de confusion au niveau des 7 départements métier (agrégation hiérarchique) :
    # mesure le taux de "sauvetage" quand l'intention prédite est fausse mais le département
    # de routage reste correct (cf. limite documentée dans le README section 4).
    department_order = sorted(set(DEPARTMENT_MAPPING.values()))
    dept_to_idx = {dept: i for i, dept in enumerate(department_order)}
    true_depts = [dept_to_idx[DEPARTMENT_MAPPING[id2label[int(label)]]] for label in test_labels]
    pred_depts = [dept_to_idx[DEPARTMENT_MAPPING[id2label[int(pred)]]] for pred in test_preds]
    cm_departments = confusion_matrix(true_depts, pred_depts, labels=list(range(len(department_order))))
    department_accuracy = accuracy_score(true_depts, pred_depts)

    print(f"  - Accuracy département (routage hiérarchique) : {department_accuracy * 100:.2f}%")

    dl_results = {
        "model_name": model_name,
        "test_accuracy": float(dl_acc),
        "test_macro_f1": float(dl_f1_macro),
        "test_weighted_f1": float(dl_f1_weighted),
        "department_accuracy": float(department_accuracy),
        "total_train_time_sec": float(total_train_time),
        "avg_latency_ms": float((eval_duration / len(test_labels)) * 1000)
    }

    confusion_data = {
        "intents": {
            "labels": [id2label[i] for i in range(len(id2label))],
            "matrix": cm_intents.tolist()
        },
        "departments": {
            "labels": department_order,
            "matrix": cm_departments.tolist()
        }
    }

    return model, tokenizer, dl_results, trainer, confusion_data


def export_artifacts(model, tokenizer, id2label, label2id, baselines_results, dl_results, confusion_data=None, output_dir="./model_artifact"):
    print("\n" + "=" * 60)
    print(f"6. EXPORT DES ARTÉFACTS DE PRODUCTION ({output_dir})")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)

    
    # 1) Sauvegarde du modèle et du tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f" Modèle et Tokenizer sauvegardés dans {output_dir}")
    
    # 2) Sauvegarde du mapping des intentions et des départements
    intent_metadata = {
        "total_intents": len(id2label),
        "total_departments": len(DEPARTMENT_DESCRIPTIONS),
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": {v: k for k, v in id2label.items()},
        "department_mapping": DEPARTMENT_MAPPING,
        "department_descriptions": DEPARTMENT_DESCRIPTIONS,
        "created_at": datetime.now().isoformat()
    }
    
    mapping_path = os.path.join(output_dir, "intent_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(intent_metadata, f, indent=2, ensure_ascii=False)
    print(f" Mapping hiérarchique sauvegardé dans {mapping_path}")
    
    # 3) Sauvegarde du rapport comparatif des métriques
    comparison_summary = {
        "project": "BankRoute AI",
        "dataset": "banking77 (mteb/banking77)",
        "num_classes": len(id2label),
        "metrics_comparison": {
            "Majority_Baseline": baselines_results["dummy"],
            "TFIDF_LogReg_Baseline": baselines_results["tfidf_logreg"],
            "DistilBERT_FineTuned": dl_results
        },
        "gain_over_majority_accuracy": f"+{(dl_results['test_accuracy'] - baselines_results['dummy']['accuracy'])*100:.2f}%",
        "gain_over_tfidf_accuracy": f"+{(dl_results['test_accuracy'] - baselines_results['tfidf_logreg']['accuracy'])*100:.2f}%",
        "gain_over_tfidf_macro_f1": f"+{(dl_results['test_macro_f1'] - baselines_results['tfidf_logreg']['macro_f1']):.4f}"
    }
    
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2, ensure_ascii=False)
    print(f" Résumé des métriques sauvegardé dans {metrics_path}")

    # 4) Sauvegarde des matrices de confusion (intentions + départements)
    if confusion_data is not None:
        confusion_path = os.path.join(output_dir, "confusion_matrices.json")
        with open(confusion_path, "w", encoding="utf-8") as f:
            json.dump(confusion_data, f, indent=2, ensure_ascii=False)
        print(f" Matrices de confusion (77 intentions + 7 départements) sauvegardées dans {confusion_path}")

    print("\n Export complet terminé avec succès !")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BankRoute AI Training Pipeline")
    parser.add_argument("--full_finetune", action="store_true", help="Désactiver LoRA et faire un Full Fine-Tuning (par défaut : LoRA/PEFT)")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epochs")
    args = parser.parse_args()

    use_lora = not args.full_finetune

    device = check_environment()
    train_data, val_data, test_data, id2label, label2id, unique_intents = load_and_prepare_data()
    baselines_results = train_baselines(train_data, val_data, test_data)
    
    if args.epochs is not None:
        epochs = args.epochs
    else:

        epochs = 2 if not torch.cuda.is_available() else 10
        
    model, tokenizer, dl_results, trainer, confusion_data = train_deep_learning_model(
        train_data, val_data, test_data, id2label, label2id,
        model_name="distilbert-base-uncased",
        num_epochs=epochs,
        use_lora=use_lora
    )
    export_artifacts(model, tokenizer, id2label, label2id, baselines_results, dl_results, confusion_data)


if __name__ == "__main__":
    main()
