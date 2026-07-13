from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from image_service import image_to_vector, preprocessing_metadata
from model_service import list_saved_models, predict_with_saved_model


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/models")
def models():
    return jsonify([model.to_dict() for model in list_saved_models()])


@app.get("/api/preprocessing")
def preprocessing():
    return jsonify(preprocessing_metadata())


@app.post("/api/predict")
def predict():
    model_filename = request.form.get("model")
    image = request.files.get("image")

    if not model_filename:
        return jsonify({"error": "Champ 'model' manquant."}), 400

    if image is None:
        return jsonify({"error": "Champ fichier 'image' manquant."}), 400

    image_vector = image_to_vector(image.stream)

    try:
        prediction = predict_with_saved_model(model_filename, image_vector)
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Erreur pendant la prédiction : {error}"}), 500

    prediction["preprocessing"] = preprocessing_metadata()
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
