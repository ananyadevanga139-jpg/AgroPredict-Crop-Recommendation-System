import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam


# ===============================
# DATASET PATH
# ===============================

train_dir = "dataset/PlantVillage/train"

val_dir = "dataset/PlantVillage/val"


# Check folders

if not os.path.exists(train_dir):
    print("Training folder not found!")
    print("Expected:", train_dir)
    exit()


if not os.path.exists(val_dir):
    print("Validation folder not found!")
    print("Expected:", val_dir)
    exit()



# ===============================
# IMAGE PREPROCESSING
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)


val_datagen = ImageDataGenerator(
    rescale=1./255
)



# ===============================
# LOAD DATASET
# ===============================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)



print("\nClasses:")
print(train_generator.class_indices)


num_classes = train_generator.num_classes


# ===============================
# MOBILE NET V2 MODEL
# ===============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


# Freeze pretrained layers

base_model.trainable = False



# Add custom layers

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)


output = Dense(
    num_classes,
    activation="softmax"
)(x)



model = Model(
    inputs=base_model.input,
    outputs=output
)



# ===============================
# COMPILE MODEL
# ===============================

model.compile(
    optimizer=Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



model.summary()



# ===============================
# TRAIN MODEL
# ===============================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)



# ===============================
# SAVE MODEL
# ===============================

if not os.path.exists("models"):
    os.makedirs("models")


model.save(
    "models/disease_model.keras"
)


print("\n==============================")
print("Disease Model Training Done!")
print("Saved as:")
print("models/disease_model.keras")
print("==============================")