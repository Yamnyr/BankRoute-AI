"""
BankRoute AI - Test Suite & Client de Preuve HTTP
Conforme aux exigences du sujet MLDL M2 :
- Appels HTTP réels avec httpx
- Tests nominatifs de routage métier
- Tests par lot
- Tests stricts des rejets 422 (validation Pydantic)
- Benchmark de latence
"""

import sys
import time
import httpx

API_BASE_URL = "http://127.0.0.1:8000"


def run_tests():
    print("=" * 70)
    print("BANKROUTE AI · SUITE DE TESTS HTTP & VALIDATION DU SERVICE")
    print(f"Cible : {API_BASE_URL}")
    print("=" * 70)

    client = httpx.Client(base_url=API_BASE_URL, timeout=30.0)
    passed_tests = 0
    total_tests = 0


    # -------------------------------------------------------------
    # TEST 1 : Health Check & Métadonnées
    # ---------------------------------------------------------------


    total_tests += 1
    print("\n[TEST 1] Vérification de l'état de santé (/health)")
    try:
        response = client.get("/health")
        assert response.status_code == 200, f"Attendu 200, reçu {response.status_code}"
        data = response.json()
        assert data["status"] == "healthy", f"Statut non healthy : {data}"
        print(f"  Statut HTTP    : {response.status_code} OK")
        print(f"  Modèle chargé  : {data.get('model_name')}")
        print(f"  Périphérique   : {data.get('device')}")
        print(f"  Classes gérées : {data.get('total_intents')} intentions dans {data.get('total_departments')} départements")
        passed_tests += 1
    except Exception as e:
        print(f" ÉCHEC Test 1 : {e}")


    # --------------------------------------------------------------
    # TEST 2 : Catalogue des intentions (/intents)
    # -------------------------------------------------------------


    total_tests += 1
    print("\n[TEST 2] Consultation du catalogue des intentions (/intents)")
    try:
        response = client.get("/intents")
        assert response.status_code == 200
        data = response.json()
        assert data["total_intents"] >= 77, f"Attendu >= 77 intentions, reçu {data['total_intents']}"
        assert data["total_departments"] == 7, f"Attendu 7 départements, reçu {data['total_departments']}"
        print(f"  Statut HTTP    : {response.status_code} OK")
        print(f"  Départements   : {list(data['departments'].keys())}")
        passed_tests += 1
    except Exception as e:
        print(f" ÉCHEC Test 2 : {e}")

    # -------------------------------------------------------------
    # TEST 3 : Prédictions métiers nominales (POST /predict)
    # -------------------------------------------------------------

    nominal_queries = [
        {
            "text": "I lost my credit card while traveling abroad, how can I block it immediately?",
            "expected_dept": "Cards_Management",
            "scenario": "Perte / Vol de carte bancaire"
        },
        {
            "text": "Can you help me cancel a pending wire transfer to another account?",
            "expected_dept": "Payments_Transfers",
            "scenario": "Annulation de virement bancaire"
        },
        {
            "text": "I forgot my account passcode and cannot log into my mobile app.",
            "expected_dept": "Account_Profile",
            "scenario": "Oubli de mot de passe / Sécurité compte"
        },
        {
            "text": "How do I activate and link Apple Pay on my new phone?",
            "expected_dept": "Digital_Services",
            "scenario": "Activation Apple Pay / Paiement mobile"
        },
        {
            "text": "The ATM machine gave me less cash than what was debited from my account.",
            "expected_dept": "Cash_ATM",
            "scenario": "Incident distributeur de billets (ATM)"
        },
        {
            "text": "Why is there an unexpected extra fee charged on my statement for currency exchange?",
            "expected_dept": "Currency_Exchange",
            "scenario": "Frais de change / Litige relevé de compte"
        }
    ]

    print("\n[TEST 3] Validation des scénarios métiers nominaux (POST /predict)")
    for i, item in enumerate(nominal_queries, 1):
        total_tests += 1
        try:
            payload = {"text": item["text"], "top_k": 3}
            t0 = time.perf_counter()
            response = client.post("/predict", json=payload)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            
            assert response.status_code == 200, f"Attendu 200, reçu {response.status_code}: {response.text}"
            res = response.json()
            assert res["predicted_intent"] is not None
            assert res["confidence"] > 0.0
            assert len(res["top_candidates"]) == 3
            assert res["department"] == item["expected_dept"], (
                f"Département attendu '{item['expected_dept']}', reçu '{res['department']}' "
                f"(intention prédite : {res['predicted_intent']})"
            )

            print(f"  [Cas {i}] Scénario : {item['scenario']}")
            print(f"         Requête   : \"{item['text'][:55]}...\"")
            print(f"         Intention : {res['predicted_intent']} (confiance: {res['confidence']*100:.1f}%)")
            print(f"         Routage   : Département {res['department']}")
            print(f"         Latence   : {lat_ms:.2f} ms")
            passed_tests += 1
        except Exception as e:
            print(f"  ÉCHEC Cas {i} : {e}")


    # -------------------------------------------------------------
    # TEST 4 : Traitement par lot (POST /predict/batch)

    # -------------------------------------------------------------

    total_tests += 1
    print("\n[TEST 4] Validation du traitement par lot (POST /predict/batch)")
    try:
        batch_payload = {
            "queries": [
                "My contactless card payment was declined at the grocery store.",
                "How long does an international bank transfer usually take?",
                "I would like to change my personal phone number in my profile."
            ],
            "top_k": 2
        }
        t0 = time.perf_counter()
        response = client.post("/predict/batch", json=batch_payload)
        batch_lat_ms = (time.perf_counter() - t0) * 1000.0
        
        assert response.status_code == 200, f"Attendu 200, reçu {response.status_code}"
        data = response.json()
        assert data["total_queries"] == 3
        assert len(data["predictions"]) == 3
        print(f"  Statut HTTP       : {response.status_code} OK")
        print(f"  Total requêtes    : {data['total_queries']}")
        print(f"  Temps global      : {batch_lat_ms:.2f} ms (moyenne {(batch_lat_ms/3):.2f} ms/requête)")
        passed_tests += 1
    except Exception as e:
        print(f" ÉCHEC Test 4 : {e}")


    # -------------------------------------------------------------
    # TEST 5 : Rejets stricts 422 (Validation Pydantic)
    # -------------------------------------------------------------

    invalid_cases = [
        {
            "name": "Texte vide",
            "payload": {"text": ""},
            "expected_code": 422
        },
        {
            "name": "Espaces uniquement",
            "payload": {"text": "      "},
            "expected_code": 422
        },
        {
            "name": "Champ 'text' manquant",
            "payload": {"top_k": 3},
            "expected_code": 422
        },
        {
            "name": "Type de donnée erroné (entier au lieu de chaîne)",
            "payload": {"text": 1234567},
            "expected_code": 422
        },
        {
            "name": "Valeur top_k hors limite (< 1)",
            "payload": {"text": "Valid inquiry text", "top_k": 0},
            "expected_code": 422
        },
        {
            "name": "Valeur top_k hors limite (> 10)",
            "payload": {"text": "Valid inquiry text", "top_k": 99},
            "expected_code": 422
        }
    ]

    print("\n[TEST 5] Validation des rejets d'erreurs 422 (Exigence non négociable)")
    for i, inv in enumerate(invalid_cases, 1):
        total_tests += 1
        try:
            response = client.post("/predict", json=inv["payload"])
            assert response.status_code == inv["expected_code"], (
                f"Attendu {inv['expected_code']}, reçu {response.status_code} pour le payload {inv['payload']}"
            )
            print(f"  [422-{i}] {inv['name']:<50} -> Rejeté avec code {response.status_code} 422 Unprocessable Entity")
            passed_tests += 1
        except Exception as e:
            print(f"  ÉCHEC Rejet 422-{i} : {e}")


    # -------------------------------------------------------------
    # TEST 6 : Benchmark de latence (p50, p95)
    # -------------------------------------------------------------

    total_tests += 1
    print("\n[TEST 6] Benchmark de performance & latence d'inférence")
    try:
        latencies = []
        bench_text = "I need assistance regarding a duplicate charge on my credit card."
        for _ in range(15):
            t_start = time.perf_counter()
            r = client.post("/predict", json={"text": bench_text, "top_k": 1})
            assert r.status_code == 200
            latencies.append((time.perf_counter() - t_start) * 1000.0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)
        print(f"  Échantillon : {len(latencies)} requêtes consécutives")
        print(f"  Latence médiane (p50) : {p50:.2f} ms")
        print(f"  Latence 95e pct (p95) : {p95:.2f} ms")
        print(f"  Latence moyenne       : {avg:.2f} ms")
        passed_tests += 1
    except Exception as e:
        print(f" ÉCHEC Test 6 : {e}")



    # -------------------------------------------------------------
    # BILAN FINAL
    # -------------------------------------------------------------

    
    print("\n" + "=" * 70)
    print(f"RÉSULTAT DES TESTS : {passed_tests} / {total_tests} validés ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 70)

    if passed_tests == total_tests:
        print(" TOUS LES TESTS SONT AU VERT ! Le service répond aux exigences du barème.")
        return 0
    else:
        print(" DES ANOMALIES ONT ÉTÉ DÉTECTÉES.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
