# API Sentiment IA — APIE638 Jour 2

API Flask d'analyse de sentiment combinant un modèle Sklearn local et l'API HuggingFace.

## Routes

| Méthode | Route        | Description                              |
|---------|--------------|------------------------------------------|
| GET     | `/health`    | Healthcheck                              |
| POST    | `/predict`   | Prédiction locale (Sklearn)              |
| POST    | `/analyse`   | Analyse via HuggingFace Inference API    |
| POST    | `/compare`   | Comparaison des deux sources             |
| GET     | `/apidocs`   | Documentation Swagger interactive        |

## Installation locale

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_votreTOKEN
python app_v2.py
# → http://localhost:5000/apidocs
```

## Déploiement sur Render

1. `git init && git add . && git commit -m "init"`
2. Pousser sur GitHub
3. render.com → **New Web Service** → connecter le dépôt
4. Build Command : `pip install -r requirements.txt`
5. Start Command : `gunicorn app_v2:app`
6. Variables d'environnement : `HF_TOKEN=hf_xxx`
7. Cliquer **Deploy**

## Déploiement Docker

```bash
docker build -t api-ia:1.0 .
docker run -p 5000:5000 -e HF_TOKEN=$HF_TOKEN api-ia:1.0
```

## Exemple d'appel /compare

```bash
curl -X POST https://votre-app.onrender.com/compare \
  -H "Content-Type: application/json" \
  -d '{"texte": "Produit excellent !"}'
```

Réponse :
```json
{
  "texte": "Produit excellent !",
  "sklearn":     {"prediction": "positif", "confiance": 0.77},
  "huggingface": {"prediction": "positif", "confiance": 0.97, "disponible": true},
  "accord": true
}
```

## Variables d'environnement

| Variable     | Obligatoire | Description                     |
|--------------|-------------|---------------------------------|
| `HF_TOKEN`   | Oui         | Token HuggingFace (hf_xxx)      |
| `MODEL_PATH` | Non         | Chemin du .pkl (défaut: local)  |
| `PORT`       | Non         | Port HTTP (injecté par Render)  |
