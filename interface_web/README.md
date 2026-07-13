# Interface web

Interface locale pour utiliser les modèles entraînés du projet.

Objectif :

- garder l'interface séparée du code des modèles ;
- servir une page HTML/CSS/JS classique ;
- exposer une petite API Python pour lire les modèles sauvegardés dans `project/PythonProject/save_model` ;
- appliquer le même preprocessing que les notebooks ;
- calculer l'inférence à partir de l'état sauvegardé dans les JSON, sans modifier les fichiers existants.

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

État actuel :

- linéaire : supporté ;
- RBF : supporté ;
- MLP : supporté avec le format JSON actuel (`weights` plat + `parameters.layer_sizes`).

Note : la DLL Rust présente localement est plus ancienne que `rust_bridge.py` et ne contient pas toutes les fonctions déclarées. L'interface lit donc directement les poids/clusters sauvegardés dans les JSON pour faire l'inférence, ce qui suffit pour afficher les modèles entraînés sans toucher au code existant.
