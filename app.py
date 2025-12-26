import os
import cv2
import numpy as np

# Initialize data dictionaries for splits
data_dict = {"Train": [], "Valid": [], "Test": []}
labels_dict = {"Train": [], "Valid": [], "Test": []}

# Loop over folds
for fold in os.listdir(dataset_path):
    fold_path = os.path.join(dataset_path, fold)
    if not os.path.isdir(fold_path):
        continue

    for split in ["Train", "Valid", "Test"]:
        split_path = os.path.join(fold_path, split)
        if not os.path.exists(split_path):
            print(f" Missing split folder: {split_path}")
            continue

        for cls in classes:
            class_path = os.path.join(split_path, cls)
            if not os.path.exists(class_path):
                print(f" Missing class folder: {class_path}")
                continue

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    data_dict[split].append(img.flatten()/255.0)
                    labels_dict[split].append(cls)

# Convert to numpy arrays
X_train = np.array(data_dict["Train"])
y_train = np.array(labels_dict["Train"])
X_val   = np.array(data_dict["Valid"])
y_val   = np.array(labels_dict["Valid"])
X_test  = np.array(data_dict["Test"])
y_test  = np.array(labels_dict["Test"])

print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train), num_classes=len(classes))
y_val   = to_categorical(le.transform(y_val), num_classes=len(classes))
y_test  = to_categorical(le.transform(y_test), num_classes=len(classes))
from sklearn.utils import class_weight
import numpy as np

y_int = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_int),
    y=y_int
)
class_weights = dict(enumerate(weights))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(512, activation="relu", input_shape=(img_size*img_size,)),
    Dense(256, activation="relu"),
    Dense(len(classes), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=100,
    class_weight=class_weights
)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
