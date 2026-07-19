from pathlib import Path


# Chemins backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PROJECT_DIR = PROJECT_ROOT / "project" / "PythonProject"
SAVE_MODEL_DIR = PYTHON_PROJECT_DIR / "save_model"

# Preprocessing modèle
CLASS_NAMES = ("FPS", "METROIDVANIA", "MOBA")
DEFAULT_IMAGE_SIZE = (71, 40)
DEFAULT_GRAYSCALE = True
NORMALIZATION_LABEL = "[-1, 1]"
