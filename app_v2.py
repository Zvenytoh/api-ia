import os
import time
import requests as req
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger

app = Flask(__name__)
CORS(app)

swagger_config = {
    "headers": [],
    "specs": [{
        "endpoint":     "apispec",
        "route":        "/apispec.json",
        "rule_filter":  lambda rule: True,
        "model_filter": lambda tag:  True,
    }],
    "static_url_path": "/flasgger_static",
    "swagger_ui":  True,
    "specs_route": "/apidocs",
}
swagger_template = {
    "info": {
        "title":       "API Sentiment IA",
        "description": (
            "Analyse de sentiment via modèle local Sklearn "
            "ou HuggingFace Inference API. "
            "Route /compare appelle les deux sources simultanément."
        ),
        "version": "2.1.0",
        "contact": {
            "name":  "E. NGOM",
            "email": "elhadji.ngomt@omicron-ailabs.com",
        },
    }
}
swagger = Swagger(app, config=swagger_config, template=swagger_template)


MODEL_PATH = os.environ.get("MODEL_PATH", "modele_sentiment.pkl")
pipeline   = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
HF_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment"
HF_URL    = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
LABEL_MAP = {
    "LABEL_0": "negatif", "LABEL_1": "neutre", "LABEL_2": "positif",
    "negative": "negatif", "neutral":  "neutre", "positive": "positif",
}


def _appel_hf(texte: str) -> dict:
    """Appel interne à l'API HuggingFace (sans retry — géré par la route)."""
    r = req.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texte},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API HF {r.status_code}: {r.text[:200]}")
    scores = r.json()
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    best = max(scores, key=lambda x: x["score"])
    return {
        "prediction": LABEL_MAP.get(best["label"], best["label"]),
        "confiance":  round(best["score"], 4),
    }


# ===============================================================
# Routes
# ===============================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Healthcheck de l'API.
    ---
    tags: [Statut]
    responses:
      200:
        description: Service opérationnel
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            hf_token_set:
              type: boolean
              example: true
            modele:
              type: string
              example: modele_sentiment.pkl
    """
    return jsonify({
        "status":       "ok",
        "hf_token_set": bool(HF_TOKEN),
        "modele":       MODEL_PATH,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prédire le sentiment via le modèle Sklearn local.
    ---
    tags: [Prédiction locale]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [texte]
          properties:
            texte:
              type: string
              example: "Ce produit est excellent !"
    responses:
      200:
        description: Prédiction réussie
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: positif
            confiance:
              type: number
              example: 0.89
            source:
              type: string
              example: sklearn_local
      400:
        description: Champ 'texte' manquant
      422:
        description: Texte trop court (min 3 caractères)
    """
    data = request.get_json(silent=True)
    if not data or "texte" not in data:
        return jsonify({"erreur": "Champ 'texte' requis"}), 400
    texte = data["texte"]
    if len(texte.strip()) < 3:
        return jsonify({"erreur": "Texte trop court (min 3 caractères)"}), 422
    try:
        pred  = pipeline.predict([texte])[0]
        proba = float(max(pipeline.predict_proba([texte])[0]))
        return jsonify({
            "prediction": pred,
            "confiance":  round(proba, 4),
            "source":     "sklearn_local",
        }), 200
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


@app.route("/analyse", methods=["POST"])
def analyse_hf():
    """
    Analyse de sentiment via HuggingFace Inference API.
    ---
    tags: [API Externe]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [texte]
          properties:
            texte:
              type: string
              example: "This product is amazing!"
    responses:
      200:
        description: Analyse réussie
        schema:
          type: object
          properties:
            texte:
              type: string
            prediction:
              type: string
              example: positif
            confiance:
              type: number
              example: 0.97
            source:
              type: string
              example: huggingface
            modele:
              type: string
              example: cardiffnlp/twitter-roberta-base-sentiment
      400:
        description: Champ 'texte' manquant
      502:
        description: Erreur API HuggingFace (code inattendu)
      503:
        description: HF_TOKEN non configuré
      504:
        description: Timeout API HuggingFace (>30s)
    """
    if not HF_TOKEN:
        return jsonify({"erreur": "HF_TOKEN non configuré"}), 503
    data = request.get_json(silent=True)
    if not data or "texte" not in data:
        return jsonify({"erreur": "Champ 'texte' requis"}), 400
    try:
        result = _appel_hf(data["texte"])
        return jsonify({
            "texte":      data["texte"],
            "prediction": result["prediction"],
            "confiance":  result["confiance"],
            "source":     "huggingface",
            "modele":     HF_MODEL,
        }), 200
    except req.Timeout:
        return jsonify({"erreur": "Timeout API HuggingFace"}), 504
    except Exception as e:
        return jsonify({"erreur": str(e)}), 502


# ---------------------------------------------------------------
# Exercice 2 & 3 — Route /compare avec documentation Swagger
# ---------------------------------------------------------------

@app.route("/compare", methods=["POST"])
def compare():
    """
    Comparer les prédictions du modèle local et de HuggingFace.

    Appelle simultanément le modèle Sklearn local et l'API
    HuggingFace, puis retourne les deux résultats côte à côte
    pour permettre une comparaison directe.
    ---
    tags: [Comparaison]
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [texte]
          properties:
            texte:
              type: string
              description: Texte à analyser (min 3 caractères)
              example: "Ce produit est vraiment fantastique !"
    responses:
      200:
        description: Comparaison réussie — les deux sources ont répondu
        schema:
          type: object
          properties:
            texte:
              type: string
              example: "Ce produit est vraiment fantastique !"
            sklearn:
              type: object
              properties:
                prediction:
                  type: string
                  example: positif
                confiance:
                  type: number
                  example: 0.89
            huggingface:
              type: object
              properties:
                prediction:
                  type: string
                  example: positif
                confiance:
                  type: number
                  example: 0.97
                disponible:
                  type: boolean
                  example: true
            accord:
              type: boolean
              description: true si les deux modèles donnent la même prédiction
              example: true
      400:
        description: |
          Champ 'texte' manquant ou texte trop court.
          Exemple de réponse : {"erreur": "Champ 'texte' requis"}
        schema:
          type: object
          properties:
            erreur:
              type: string
              example: "Champ 'texte' requis"
      502:
        description: |
          Erreur lors de l'appel à l'API HuggingFace.
          La réponse sklearn reste disponible dans le corps.
        schema:
          type: object
          properties:
            texte:
              type: string
            sklearn:
              type: object
              properties:
                prediction:
                  type: string
                confiance:
                  type: number
            huggingface:
              type: object
              properties:
                disponible:
                  type: boolean
                  example: false
                erreur:
                  type: string
                  example: "HF_TOKEN non configuré"
    """
    data = request.get_json(silent=True)
    if not data or "texte" not in data:
        return jsonify({"erreur": "Champ 'texte' requis"}), 400

    texte = data["texte"]
    if len(texte.strip()) < 3:
        return jsonify({"erreur": "Texte trop court (min 3 caractères)"}), 400

    # --- Prédiction locale (toujours disponible) ---
    pred_local  = pipeline.predict([texte])[0]
    proba_local = float(max(pipeline.predict_proba([texte])[0]))
    sklearn_result = {
        "prediction": pred_local,
        "confiance":  round(proba_local, 4),
    }

    # --- Prédiction HuggingFace (optionnelle — dégradation gracieuse) ---
    if not HF_TOKEN:
        hf_result  = {"disponible": False, "erreur": "HF_TOKEN non configuré"}
        http_code  = 502
    else:
        try:
            r         = _appel_hf(texte)
            hf_result = {"disponible": True, **r}
            http_code = 200
        except req.Timeout:
            hf_result = {"disponible": False, "erreur": "Timeout API HuggingFace"}
            http_code = 502
        except Exception as e:
            hf_result = {"disponible": False, "erreur": str(e)}
            http_code = 502

    # accord = True si les deux sources sont disponibles et concordent
    accord = (
        hf_result.get("disponible", False)
        and hf_result.get("prediction") == sklearn_result["prediction"]
    )

    return jsonify({
        "texte":       texte,
        "sklearn":     sklearn_result,
        "huggingface": hf_result,
        "accord":      accord,
    }), http_code


@app.errorhandler(404)
def not_found(e):
    return jsonify({"erreur": "Route introuvable", "code": 404}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
