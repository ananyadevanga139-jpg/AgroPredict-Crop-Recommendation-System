import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset

data = pd.read_csv(
    "datasets/crop_recommendation.csv"
)


print(data.head())


# Separate input and output

X = data.drop("label", axis=1)

y = data["label"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train

model.fit(
    X_train,
    y_train
)


# Prediction

prediction = model.predict(X_test)


# Accuracy

accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Model Accuracy:",
    accuracy*100,
    "%"
)


# Save model

with open(
    "models/crop_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("Model Saved Successfully!")