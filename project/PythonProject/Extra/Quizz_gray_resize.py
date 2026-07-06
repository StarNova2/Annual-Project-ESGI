from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# -------------------------
# PARAMETRES
# -------------------------

DATASET_PATH = Path(input("Give the Path to the dataset : "))
if DATASET_PATH == "":
    print("No dataset given, reverting to default")
    DATASET_PATH = Path(r"C:/Users/theot/Pictures/Dataset")


TAILLE_Y = 40
TAILLE_X = int(TAILLE_Y * 16/9)

NB_IMAGES_PAR_CLASSE = 2

# -------------------------
# RECUPERATION DES IMAGES
# -------------------------

images = []

for dossier in DATASET_PATH.iterdir():

    if not dossier.is_dir():
        continue

    fichiers = list(dossier.glob("*.png"))

    if len(fichiers) == 0:
        continue

    selection = random.sample(
        fichiers,
        min(NB_IMAGES_PAR_CLASSE, len(fichiers))
    )

    for fichier in selection:
        images.append((dossier.name, fichier))


# Mélange du quiz
random.shuffle(images)


# -------------------------
# QUIZ
# -------------------------

index = 0
reponse_visible = False


fig, ax = plt.subplots(figsize=(5, 5))


def charger_image():
    global index, reponse_visible

    ax.clear()

    classe, chemin = images[index]

    img = Image.open(chemin)
    img = img.convert("L")
    img = img.resize((TAILLE_X, TAILLE_Y))

    ax.imshow(img)#, cmap="gray"

    ax.axis("off")

    if reponse_visible:
        ax.set_title(f"Réponse : {classe}", fontsize=14)
    else:
        ax.set_title(
            f"Question {index+1}/{len(images)}\n"
            "Devine la classe",
            fontsize=14
        )

    plt.draw()


def on_key(event):
    global index, reponse_visible

    # ESPACE = montrer la réponse
    if event.key == " ":
        reponse_visible = True
        charger_image()


    # ENTER = image suivante
    elif event.key == "enter":

        index += 1

        if index >= len(images):
            print("Quiz terminé !")
            plt.close()
            return

        reponse_visible = False
        charger_image()


fig.canvas.mpl_connect("key_press_event", on_key)


charger_image()

plt.show()


# -------------------------
# TEST DU VECTEUR
# -------------------------

classe, chemin = random.choice(images)

img = Image.open(chemin)
img = img.convert("L")
img = img.resize((TAILLE_X, TAILLE_Y))

X = np.array(img, dtype=np.float32)

X = X.flatten()
X /= 255.0

print("Image test :", chemin.name)
print("Classe réelle :", classe)
print("Shape image :", img.size)
print("Shape vecteur :", X.shape)
print("Nombre de features :", len(X))