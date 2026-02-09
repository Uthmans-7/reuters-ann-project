Reuters News Classifier AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---
##  Project Overview
This is a production-ready AI service capable of classifying Reuters news articles into **46 distinct topics** (e.g., Earnings, Acquisitions, Crude Oil). 

It features a **Deep Learning (ANN)** backend, a secure **REST API**, and a user-friendly **Web Interface** for real-time inference.


## Key Features
* **AI Engine:** Custom-trained Artificial Neural Network (ANN) achieving high accuracy on the Reuters-21578 dataset.
* **Secure API:** Protected via **HTTP Basic Authentication** to prevent unauthorized access.
* **Interactive UI:** A built-in web tool for manual testing and demonstration.
* **Health Monitoring:** Dedicated `/health` endpoint for system status checks (Liveness Probe).
* **Data Validation:** Strict Pydantic schemas ensure robust input handling.


##  Installation & Setup

### 1. Clone the Repository
git clone [https://github.com/Uthmans-7/reuters-ann-project.git](https://github.com/Uthmans-7/reuters-ann-project.git)
cd reuters-ann-project

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run the Server

uvicorn api:app --reload


## API Documentation

| Endpoint | Method | Description | Auth Required |
| --- | --- | --- | --- |
| `/` | `GET` | System Home Page | ❌ No |
| `/predict-ui` | `GET` | **Interactive Analysis Tool** (Web Interface) | ✅ **Yes** |
| `/predict` | `POST` | **Inference API** (Returns JSON prediction) | ✅ **Yes** |
| `/health` | `GET` | System Health Check (for monitoring) | ❌ No |
| `/docs` | `GET` | Swagger UI Documentation | ❌ No |



## Access Credentials

The system is protected. Use these credentials to access the UI and API:

* **Username:** `admin`
* **Password:** `password123`


## Model Architecture

* **Input Layer:** Vectorized text (10,000 dimensions).
* **Hidden Layers:** Dense layers with ReLU activation and Dropout for regularization.
* **Output Layer:** Softmax layer predicting probabilities across 46 classes.
* **Optimizer:** Adam.
* **Loss Function:** Sparse Categorical Crossentropy.



## Project Structure

reuters-ann-project/
├── api.py              # Main FastAPI application (Endpoints & Security)
├── src/
│   ├── data_loader.py  # Data preprocessing logic
│   ├── model_builder.py # Neural Network architecture
│   └── trainer.py      # Training loop & experimentation
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation



**Author:** AI Engineering Team

