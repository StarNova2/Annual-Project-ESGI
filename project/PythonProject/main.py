import numpy as np
from dataset_loader import load_labeled_image_dataset, stratified_split
from project.PythonProject.rust_bridge import OVRLinearClassifier


def confusion_matrix(y_true, y_pred, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int32)

    true_indices = np.argmax(y_true, axis=1)
    pred_indices = np.argmax(y_pred, axis=1)

    for true_idx, pred_idx in zip(true_indices, pred_indices):
        matrix[true_idx, pred_idx] += 1

    return matrix
print("hi")

dataset = load_labeled_image_dataset(
    image_size=(20,32),
    grayscale=True
)


split = stratified_split(
    dataset,
    test_ratio=0.2
)


print("Train :", split.x_train.shape)
print("Test :", split.x_test.shape)


model = OVRLinearClassifier(
    input_dim=split.x_train.shape[1],
    output_dim=3,
    learning_rate=0.01
)


model.fit(
    split.x_train,
    split.y_train,
    epochs=20000
)


prediction = model.predict_labels(split.x_test)

matrice_de_confu = confusion_matrix(
    split.y_test,
    prediction,
    len(dataset.class_names)
)

print(matrice_de_confu)

print(classification_report(
    np.argmax(split.y_test, axis=1),
    np.argmax(prediction, axis=1),
    target_names=dataset.class_names
))


accuracy = (prediction == split.y_test).all(axis=1).mean()

print("Accuracy :", accuracy)


model.close()