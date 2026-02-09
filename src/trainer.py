import keras_tuner as kt
from src.model_builder import build_model
import sqlite3
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# --- HELPER: CALCULATE METRICS ---
def get_metrics(model, X_test, y_test):
    """Calculates deep metrics for a single model"""

    #First, predict class probabilities for each test sample.
    y_pred_probs = model.predict(X_test, verbose=0)
    #Then pick the class with the highest probability as the predicted label.
    y_pred = np.argmax(y_pred_probs, axis=1)
    #Convert the true one-hot labels into class labels for comparison.
    y_true = np.argmax(y_test, axis=1)
    
    # Specificity Metrics 
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_specificity = np.nanmean(tn / (tn + fp))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1_score': report['macro avg']['f1-score'],
        'specificity': avg_specificity
    }

# --- MAIN DATABASE FUNCTION ---
def save_all_trials_with_metrics(tuner, X_test, y_test):
    conn = sqlite3.connect('results/trials.db')
    cursor = conn.cursor()

    # 1. New Schema: Includes all metrics + is_best flag
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tuning_trials (
            trial_id TEXT PRIMARY KEY,
            model_name TEXT,
            is_best INTEGER,  -- 1 if best, 0 otherwise
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            specificity REAL,
            hyperparameters TEXT,
            timestamp TEXT
        )
    ''')
    
    # Optional: Clear old data to prevent ID conflicts
    # cursor.execute('DELETE FROM tuning_trials')

    print("\n---  PROCESSING ALL TRIALS (Calculated Metrics for Each) ---")
    
    # Identify the Winner ID
    best_trial_id = tuner.oracle.get_best_trials(1)[0].trial_id
    
    # Loop through ALL trials
    for trial_id, trial in tuner.oracle.trials.items():
        print(f"   > Processing Trial {trial_id}...", end=" ")
        
        try:
            # A. Load the model for this specific trial
            model = tuner.load_model(trial)
            
            # B. Get the hard numbers
            metrics = get_metrics(model, X_test, y_test)
            
            # C. Check if it's the winner
            is_best = 1 if trial_id == best_trial_id else 0
            
            # D. Prepare Data
            hyperparameters = json.dumps(trial.hyperparameters.values)
            now = datetime.now()
            model_name = f"Ann_{now.strftime('%d%m%Y_%H%M%S')}_{trial_id}"
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

            # E. Save to DB
            cursor.execute('''
                INSERT OR REPLACE INTO tuning_trials 
                (trial_id, model_name, is_best, accuracy, precision, recall, f1_score, specificity, hyperparameters, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trial_id, model_name, is_best, 
                metrics['accuracy'], metrics['precision'], metrics['recall'], 
                metrics['f1_score'], metrics['specificity'], 
                hyperparameters, timestamp
            ))
            
            tag = " WINNER" if is_best else "✅"
            print(f"{tag} (Acc: {metrics['accuracy']:.4f})")
            
        except Exception as e:
            print(f" Failed to load/eval trial {trial_id}: {e}")

    conn.commit()
    conn.close()
    print("---  DATABASE UPDATED SUCCESSFULLY ---")

# --- TUNER FUNCTION ---
def tune_and_train(X_train, y_train, X_test, y_test, num_words, num_classes, max_trials=5, epochs=10):
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, num_words, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='results',
        project_name='reuters_ann'
    )
    
    tuner.search(X_train, y_train, epochs=epochs, validation_split=0.3, batch_size=512)
    
    # CALL THE NEW SAVE FUNCTION
    save_all_trials_with_metrics(tuner, X_test, y_test)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner

# --- PLOT FUNCTION ---
def evaluate_final(model, X_test, y_test):
    """Just generates the Confusion Matrix plot for the winner"""
    print("\n--- Generating Confusion Matrix ---")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title('Best Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    print("  Saved 'confusion_matrix.png'")