import tensorflow as tf

model = tf.keras.models.load_model(
    "models/disease_model.keras",
    compile=False
)

model.save(
    "models/disease_model.keras"
)

print("Model resaved successfully")