import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Load dataset

data = pd.read_csv(
    "datasets/fertilizer.csv"
)


# Separate encoders

soil_encoder = LabelEncoder()

crop_encoder = LabelEncoder()


# Encode soil and crop

data["Soil_Type"] = soil_encoder.fit_transform(
    data["Soil_Type"]
)


data["Crop_Type"] = crop_encoder.fit_transform(
    data["Crop_Type"]
)



# Features and target

X = data.drop(
    "Fertilizer",
    axis=1
)


y = data["Fertilizer"]



# Train model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X,
    y
)



# Save model

pickle.dump(
    model,
    open(
        "models/fertilizer_model.pkl",
        "wb"
    )
)


pickle.dump(
    soil_encoder,
    open(
        "models/soil_encoder.pkl",
        "wb"
    )
)


pickle.dump(
    crop_encoder,
    open(
        "models/crop_encoder.pkl",
        "wb"
    )
)


print("Fertilizer model trained successfully")