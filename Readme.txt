
# Reuters Automated News Classifier

An AI-powered system that automatically classifies news articles into (46 distinct topics) (e.g., Earnings, Grain, Acquisitions, Crude Oil) using Deep Learning.

This project goes beyond simple training by implementing **automated hyperparameter tuning**, **SQL-based experiment tracking**, and a **FastAPI deployment** for real-time predictions.



##  Key Features

* ** Deep Learning Model:** A Multi-Layer Perceptron (ANN) optimized for text classification.
* ** Automated Tuning:** Uses **Keras Tuner** to automatically find the best network architecture (neurons, learning rate, dropout).
* ** Experiment Database:** Logs every single training trial and its metrics into a **SQLite Database** (`trials.db`) for analysis.
* ** Advanced Metrics:** Calculates **Precision, Recall, F1-Score, and Specificity** to handle class imbalance effectively.
* ** Visualization:** Automatically generates a **Confusion Matrix** to visualize model performance.
* ** Real-Time API:** A fully functional **FastAPI** server that accepts raw text and returns the predicted topic and confidence score.

---

##  Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow, Keras
* **Optimization:** Keras Tuner (Random Search)
* **Data Processing:** NumPy, Scikit-Learn
* **Deployment:** FastAPI, Uvicorn
* **Database:** SQLite3
* **Visualization:** Matplotlib, Seaborn

---

##  Project Structure


REUTERS_ANN_PROJECT/
├── src/
│   ├── data_loader.py    # Loads & vectorizes Reuters dataset (Bag-of-Words)
│   ├── model_builder.py  # Defines the Search Space for the Neural Network
│   └── trainer.py        # Handles Training, DB Logging, & Metric Calculation
├── results/
│   └── trials.db         # SQLite Database containing all experiment logs
├── api.py                # FastAPI Server for real-time predictions
├── run.py                # Main script to run the Training pipeline
├── reuters_best_model.keras  # The saved "Champion" model
├── confusion_matrix.png  # Performance visualization
└── README.md             # Project Documentation

```

---

##  How to Run

### 1. Install Dependencies


pip install tensorflow keras-tuner fastapi uvicorn scikit-learn matplotlib seaborn numpy


### 2. Train & Tune the Model

Run the main script to start the experiment. This will load data, search for the best model architecture, save the results to SQLite, and generate the confusion matrix.


python run.py


*Output:* You will see training logs and a final "Detailed Performance Report" in the terminal.

### 3. Start the API Server

Launch the backend server to serve predictions.

```bash
uvicorn api:app --reload

```

### 4. Test the API

Open your browser and navigate to the interactive documentation:
**[http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)**

---

##  Database & Metrics

The project uses a structured SQL database (`results/trials.db`) to track progress.

### Table 1: `tuning_trials`

Tracks every experiment run by the tuner.

* **`trial_id`**: Unique ID for the experiment.
* **`hyperparameters`**: JSON string of settings used (e.g., `units_layer1: 64`, `lr: 0.001`).
* **`accuracy`**: Validation accuracy of the model.
* **`is_best`**: Flag (1 or 0) indicating the "Champion" model.

### Table 2: `final_reports` (Optional View)

Contains the deep-dive metrics for the best model:

* **Specificity**: ~99% (Crucial for imbalanced multi-class problems).
* **F1-Score**: Macro-averaged score to account for minority classes.

---

## Example API Usage

**Request (JSON):**

```json
{
  "text": "Grain prices soared today as wheat production hit an all time low in the midwest"
}

```

**Response (JSON):**

```json
{
  "prediction": "GRAIN",
  "class_id": 1,
  "confidence": "16.86%",
  "original_text": "Grain prices soared..."
}

```

---

##  Author

Developed as part of the AI Internship Program.

* **Focus:** NLP, Model Optimization, and Backend Integration.
* **Date:** February 2026

