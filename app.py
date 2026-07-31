from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import os
from disease_info import disease_info


app = Flask(__name__)


# =====================================
# LOAD MODELS
# =====================================

crop_model = pickle.load(
    open("models/crop_model.pkl", "rb")
)


fertilizer_model = pickle.load(
    open("models/fertilizer_model.pkl", "rb")
)


soil_encoder = pickle.load(
    open("models/soil_encoder.pkl", "rb")
)


crop_encoder = pickle.load(
    open("models/crop_encoder.pkl", "rb")
)



# Disease Model

# Disease Model
disease_model = tf.keras.models.load_model(
    "models/disease_model.keras",
    compile=False
)


with open("disease_classes.json","r") as f:
    disease_classes = json.load(f)



# Upload folder

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER




# =====================================
# HOME
# =====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )





# =====================================
# CROP PREDICTION
# =====================================
@app.route("/crop")
def crop():

    return render_template(
        "crop.html"
    )
@app.route("/predict", methods=["POST"])
def predict():

    try:

        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))

        temperature = float(
            request.form.get("temperature")
        )

        humidity = float(
            request.form.get("humidity")
        )

        ph = float(
            request.form.get("ph")
        )

        rainfall = float(
            request.form.get("rainfall")
        )


        input_data = np.array(
            [[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]]
        )


        prediction = crop_model.predict(
            input_data
        )


        try:
            crop = crop_encoder.inverse_transform(
                prediction
            )[0]

        except:
            crop = prediction[0]



        return render_template(
            "result.html",
            crop=crop
        )


    except Exception as e:

        return f"Crop Error : {e}"






# =====================================
# FERTILIZER PAGE
# =====================================

@app.route("/fertilizer")
def fertilizer():

    return render_template(
        "fertilizer.html"
    )





# =====================================
# FERTILIZER PREDICTION
# =====================================

@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():

    try:

        temperature = float(
            request.form.get("temperature")
        )

        humidity = float(
            request.form.get("humidity")
        )

        moisture = float(
            request.form.get("moisture")
        )


        soil_type = request.form.get(
            "soil_type"
        )


        crop_type = request.form.get(
            "crop_type"
        )


        N = float(
            request.form.get("N")
        )


        P = float(
            request.form.get("P")
        )


        K = float(
            request.form.get("K")
        )



        soil_value = soil_encoder.transform(
            [soil_type]
        )[0]


        crop_value = crop_encoder.transform(
            [crop_type]
        )[0]



        fertilizer_input = np.array(
            [[
                temperature,
                humidity,
                moisture,
                soil_value,
                crop_value,
                N,
                P,
                K
            ]]
        )


        prediction = fertilizer_model.predict(
            fertilizer_input
        )


        fertilizer = prediction[0]



        return render_template(
            "fertilizer_result.html",
            fertilizer=fertilizer
        )



    except Exception as e:

        return f"Fertilizer Error : {e}"







# =====================================
# DISEASE DETECTION PAGE
# =====================================

@app.route("/disease")
def disease():

    return render_template(
        "disease.html"
    )




# =====================================
# DISEASE PREDICTION
# =====================================

@app.route("/predict_disease", methods=["POST"])
def predict_disease():

    try:

        file = request.files["image"]

        if file.filename == "":
            return "No image selected"


        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )


        file.save(filepath)



        img = Image.open(filepath).convert("RGB")


        img = img.resize(
            (224,224)
        )


        img = np.array(img) / 255.0


        img = np.expand_dims(
            img,
            axis=0
        )



        prediction = disease_model.predict(img)



        class_index = np.argmax(
            prediction
        )


        confidence = float(
            np.max(prediction)
        ) * 100



        disease_name = disease_classes[class_index]



        info = disease_info.get(
            disease_name,
            {
                "symptoms": "Information not available.",
                "treatment": "Consult an agricultural expert.",
                "prevention": "Follow good farming practices."
            }
        )



        return render_template(
            "disease_result.html",
            disease=disease_name.replace("___"," - ").replace("_"," "),
            confidence=round(confidence,2),
            image=file.filename,
            symptoms=info["symptoms"],
            treatment=info["treatment"],
            prevention=info["prevention"]
        )


    except Exception as e:

        return f"Disease Error : {e}"




# =====================================
# RUN
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )