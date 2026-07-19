# Interface web

Interface locale pour utiliser les modèles entraînés du projet.

## Structure

```text
interface_web/
  frontend/   Page web et assets statiques
  backend/    API locale Python
```

## Lancement

```powershell
cd interface_web/backend
py -m pip install -r requirements.txt
py app.py
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

## Routes

- `GET /api/models` : liste les modèles JSON sauvegardés ;
- `GET /api/preprocessing` : décrit le preprocessing appliqué aux images ;
- `POST /api/predict` : reçoit `model` + `image`, puis renvoie la classe prédite.

## Modèles

L'interface affiche les modèles JSON présents dans `project/PythonProject/save_model`.

Pour garder une démo propre, elle affiche un seul modèle par famille :

- linéaire : priorité à `linear/model_linear.json`, puis fallback vers un autre JSON linéaire valide ;
- RBF : priorité à `rbf/model_rbf.json`, puis fallback vers un autre JSON RBF valide ;
- MLP : priorité à `mlp/model_mlp.json`, puis fallback vers un autre JSON MLP valide.

Un fichier absent, vide ou invalide est ignoré automatiquement.