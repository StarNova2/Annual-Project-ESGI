# Interface web

Interface locale pour utiliser les modèles entraînés du projet.

Objectif :

- garder l'interface séparée du code des modèles ;
- servir une page HTML/CSS/JS classique ;
- exposer une petite API Python pour lire les modèles sauvegardés dans `PythonProject/save_model` ;
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

L'interface affiche les modèles JSON présents dans `PythonProject/save_model`.

État actuel :

- linéaire : supporté ;
- RBF : supporté ;
- MLP : prévu dans l'interface. La carte apparaît déjà en "à venir" tant qu'aucun JSON MLP n'est présent. Quand un JSON MLP sera ajouté, il sera listé automatiquement ; il faudra seulement confirmer le format exact des poids si la sauvegarde ne reprend pas la structure Rust native.

Note : la DLL Rust présente localement est plus ancienne que `rust_bridge.py` et ne contient pas toutes les fonctions déclarées. L'interface lit donc directement les poids/clusters sauvegardés dans les JSON pour faire l'inférence, ce qui suffit pour afficher les modèles entraînés sans toucher au code existant.
