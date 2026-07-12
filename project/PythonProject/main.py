import ctypes
import numpy as np
import matplotlib.pyplot as plt
import random

#Class MyDroite qui reprend la structure sur la lib rust
class MyDroite(ctypes.Structure):_fields_ = [
    ("a", ctypes.c_float),
    ("b", ctypes.c_float),
    ("c", ctypes.c_float),
]

def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    lib = ctypes.cdll.LoadLibrary('../lib_classification/target/release/lib_classification.dll')

    #innitialisation des 1000 points
    points = np.array([[random.random(), random.random()] for _ in range(1000)], dtype=np.float32)
    data_ptr = points.ctypes.data_as(ctypes.POINTER(ctypes.c_float))



    #Affichage
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

    #Attribution des couleurs --> les labels
    labels = []
    colors = []

    for p in points:
        if p[0] + p[1] - 0.7 >= 0:
            labels.append(1)
            colors.append('blue')
        else:
            labels.append(-1)
            colors.append('red')

    labels = np.array(labels, dtype=np.int8)

    #Affichage des données
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()

    #Innitialisation de la droite de base
    lib.initialisation_droite.restype = ctypes.POINTER(MyDroite)
    ma_droite = lib.initialisation_droite()
    a = ma_droite.contents.a
    b = ma_droite.contents.b
    c = ma_droite.contents.c
    w = MyDroite(a, b, c)


    #Assignation des couleurs
    #lib.linear_classification_prediction.restype = ctypes.c_int
    lib.linear_classification_prediction.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float
    ]

    lib.linear_classification_prediction.restype = ctypes.c_int8

    points_to_predict = []
    predicted_colors = []
    for row in range(100):
        for col in range(100):
            point_to_predict = [1.0, row / 100.0, col / 100.0]
            points_to_predict.append(point_to_predict)
            predicted_colors.append(
                'lightblue' if lib.linear_classification_prediction(w.a, w.b, w.c, point_to_predict[0], point_to_predict[1], point_to_predict[2]) >= 0 else 'pink')
    points_to_predict = np.array(points_to_predict)

    #Affichage
    plt.scatter(points_to_predict[:, 1], points_to_predict[:, 2], c=predicted_colors)
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()


    labels_ptr = labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    lib.training.restype = ctypes.POINTER(MyDroite)
    w = lib.training(10000, 1000, w, data_ptr, labels_ptr)
    w = w.contents



    #Assignation des couleurs
    points_to_predict = []
    predicted_colors = []
    print(w)
    for row in range(100):
        for col in range(100):
            point_to_predict = [1.0, row / 100.0, col / 100.0]
            points_to_predict.append(point_to_predict)
            predicted_colors.append(
                'lightblue' if lib.linear_classification_prediction(w.a, w.b, w.c, point_to_predict[0], point_to_predict[1], point_to_predict[2]) >= 0 else 'pink')
    points_to_predict = np.array(points_to_predict)


    #Affichage après entrainement
    plt.scatter(points_to_predict[:, 1], points_to_predict[:, 2], c=predicted_colors)
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()
