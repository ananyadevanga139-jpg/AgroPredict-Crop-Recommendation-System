# 🌱 AgroPredict - Crop Recommendation & Plant Disease Prediction System

AgroPredict is an AI-based agricultural assistance system that helps users make better farming decisions by recommending suitable crops, fertilizers, and detecting plant diseases using Machine Learning and Deep Learning techniques.

## 🚀 Features

* 🌾 Crop Recommendation System

  * Recommends suitable crops based on soil and environmental parameters.

* 🧪 Fertilizer Recommendation System

  * Suggests appropriate fertilizers for better crop productivity.

* 🍃 Plant Disease Detection

  * Detects plant diseases from uploaded images using a CNN-based deep learning model.

* 🌐 Web Application

  * User-friendly interface for real-time predictions.

## 🛠️ Technologies Used

* Python
* Flask
* TensorFlow
* Keras
* Scikit-learn
* Machine Learning
* Deep Learning
* HTML
* CSS
* JavaScript

## 📂 Project Structure

```
AgroPredict/
│
├── app.py
├── models/
│   ├── crop_model.pkl
│   ├── fertilizer_model.pkl
│   ├── disease_model.keras
│
├── templates/
├── static/
├── disease_info.py
├── disease_classes.json
├── requirements.txt
├── Procfile
└── runtime.txt
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AgroPredict-Crop-Recommendation-System.git
```

### 2. Navigate to project folder

```bash
cd AgroPredict-Crop-Recommendation-System
```

### 3. Create virtual environment

```bash
python -m venv venv
```

### 4. Activate environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run application

```bash
python app.py
```

Application will run at:

```
http://127.0.0.1:5000/
```

## ☁️ Deployment

The AgroPredict web application is deployed using **Render**.

Deployment steps:

1. Push project code to GitHub repository.
2. Connect GitHub repository with Render.
3. Configure Python environment.
4. Install dependencies from `requirements.txt`.
5. Deploy the Flask application using Gunicorn.

Deployment configuration:

**Procfile**

```
web: gunicorn app:app
```

**Runtime**

```
Python 3.11
```

## 📌 Model Details

### Crop Recommendation Model

* Machine Learning based classification model.
* Uses soil and environmental parameters.

### Fertilizer Recommendation Model

* Predicts suitable fertilizer recommendations.

### Disease Detection Model

* CNN-based deep learning model.
* Saved as:

```
disease_model.keras
```

## 🎯 Future Enhancements

* Add more crop and disease classes.
* Improve model accuracy with larger datasets.
* Add mobile application support.
* Integrate weather API for real-time recommendations.

## 👩‍💻 Developed By

**Ananya K**


