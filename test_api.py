"""
Script de validation de l'API Sentiment IA — APIE638 J2
Usage : python test_api.py [BASE_URL]
        python test_api.py https://api-ia.medev-tech.fr
"""

import sys
import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "https://api-ia.medev-tech.fr"

PASS = "✅"
FAIL = "❌"
results = []


def test(nom, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    print(f"  {status} {nom}" + (f" — {detail}" if detail else ""))


def section(titre):
    print(f"\n{'='*55}")
    print(f"  {titre}")
    print(f"{'='*55}")


print(f"\nCible : {BASE_URL}\n")

section("1. GET /health")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=15, verify=False)
    test("HTTP 200",          r.status_code == 200, f"code={r.status_code}")
    test("JSON valide",       r.headers.get("content-type","").startswith("application/json"))
    data = r.json()
    test("status == 'ok'",    data.get("status") == "ok")
    test("hf_token_set",      data.get("hf_token_set") is True, "HF_TOKEN configuré")
except Exception as e:
    test("Connexion", False, str(e))

section("2. POST /predict (modèle local Sklearn)")
try:
    r = requests.post(f"{BASE_URL}/predict",
                      json={"texte": "Ce produit est absolument fantastique !"},
                      timeout=15, verify=False)
    test("HTTP 200",          r.status_code == 200, f"code={r.status_code}")
    data = r.json()
    test("champ 'prediction'", "prediction" in data, data.get("prediction"))
    test("champ 'confiance'",  "confiance"  in data, str(data.get("confiance")))
    test("source == local",    data.get("source") == "sklearn_local")
    test("prédiction valide",  data.get("prediction") in ("positif","negatif","neutre"))
except Exception as e:
    test("Connexion", False, str(e))

section("3. POST /predict — gestion des erreurs")
try:
    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=10, verify=False)
    test("400 si champ manquant", r.status_code == 400, f"code={r.status_code}")

    r = requests.post(f"{BASE_URL}/predict", json={"texte": "ab"}, timeout=10, verify=False)
    test("422 si texte trop court", r.status_code == 422, f"code={r.status_code}")
except Exception as e:
    test("Erreurs /predict", False, str(e))

section("4. POST /analyse (API HuggingFace)")
try:
    r = requests.post(f"{BASE_URL}/analyse",
                      json={"texte": "This product is terrible, I want a refund!"},
                      timeout=30, verify=False)
    test("HTTP 200",           r.status_code == 200, f"code={r.status_code}")
    data = r.json()
    test("source == huggingface", data.get("source") == "huggingface")
    test("champ 'modele'",        "modele" in data, data.get("modele",""))
    test("prédiction négative",   data.get("prediction") == "negatif",
         data.get("prediction","?"))
except Exception as e:
    test("Connexion HF", False, str(e))

section("5. POST /compare (comparaison des deux sources)")
try:
    r = requests.post(f"{BASE_URL}/compare",
                      json={"texte": "Produit excellent, je recommande !"},
                      timeout=30, verify=False)
    test("HTTP 200",               r.status_code == 200, f"code={r.status_code}")
    data = r.json()
    test("champ 'sklearn'",        "sklearn"     in data)
    test("champ 'huggingface'",    "huggingface" in data)
    test("champ 'accord'",         "accord"      in data)
    test("sklearn.prediction",     "prediction" in data.get("sklearn", {}))
    test("huggingface.disponible", data.get("huggingface",{}).get("disponible") is True)

    sk  = data["sklearn"]["prediction"]
    hf  = data["huggingface"].get("prediction","?")
    print(f"\n     Sklearn     : {sk} ({data['sklearn']['confiance']:.0%})")
    print(f"     HuggingFace : {hf} ({data['huggingface'].get('confiance',0):.0%})")
    print(f"     Accord      : {data['accord']}")
except Exception as e:
    test("Connexion", False, str(e))

section("6. POST /compare — gestion des erreurs")
try:
    r = requests.post(f"{BASE_URL}/compare", json={}, timeout=10, verify=False)
    test("400 si champ manquant", r.status_code == 400, f"code={r.status_code}")
    r = requests.post(f"{BASE_URL}/compare", json={"texte": "ab"}, timeout=10, verify=False)
    test("400 si texte trop court", r.status_code == 400, f"code={r.status_code}")
except Exception as e:
    test("Erreurs /compare", False, str(e))

section("7. Route inconnue → 404")
try:
    r = requests.get(f"{BASE_URL}/route_inexistante", timeout=10, verify=False)
    test("404 retourné",  r.status_code == 404, f"code={r.status_code}")
    test("JSON en retour", r.headers.get("content-type","").startswith("application/json"))
except Exception as e:
    test("404 handler", False, str(e))

section("8. Documentation Swagger")
try:
    r = requests.get(f"{BASE_URL}/apidocs", timeout=10, verify=False)
    test("HTTP 200 /apidocs", r.status_code == 200)
    r = requests.get(f"{BASE_URL}/apispec.json", timeout=10, verify=False)
    test("HTTP 200 /apispec.json", r.status_code == 200)
    spec = r.json()
    routes = list(spec.get("paths", {}).keys())
    test("/compare documentée",  "/compare" in routes,  str(routes))
    test("/predict documentée",  "/predict" in routes)
    test("/analyse documentée",  "/analyse" in routes)
    test("/health documentée",   "/health"  in routes)
except Exception as e:
    test("Swagger", False, str(e))

total  = len(results)
passed = sum(results)
print(f"\n{'='*55}")
print(f"  RÉSULTAT : {passed}/{total} tests passés", PASS if passed == total else FAIL)
print(f"{'='*55}\n")

