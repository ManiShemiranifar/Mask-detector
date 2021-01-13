from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to dataset")

ap.add_argument("-m", "--model",
                help="model output name")
args = vars(ap.parse_args())

images_paths = list(paths.list_images(args["dataset"]))

data = []
labels = []

for image_path in images_paths:
    label = image_path.split(os.path.sep)[0]

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

data_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

mobileNet_model = MobileNetV2(weights="imagenet", include_top=False,
                              input_shape=(224, 224, 3))

head_model = mobileNet_model.output
head_model = layers.AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = layers.Flatten()(head_model)
head_model = layers.Dense(128, activation="relu")(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(2, activation="softmax")(head_model)

model = Model(inputs=mobileNet_model.input, outputs=head_model)

for layer in mobileNet_model.layers:
    layer.trainable = False

model.compile(loss="binary_crossentropy", optimizer=RMSprop(1e-4),
              metrics=["acc"])

history = model.fit(
    data_generator.flow(x_train, y_train),
    steps_per_epoch=len(x_train) // 32,
    validation_data=(x_train, y_train),
    validation_steps=len(x_train) // 32,
    epochs=20
)

pred = model.predict(x_test, batch_size=32)
pred = np.argmax(pred, axis=1)

print(classification_report(y_test.argmax(axis=1), pred,
                            target_names=lb.classes_))

model.save(args["model"], save_format="h5")
