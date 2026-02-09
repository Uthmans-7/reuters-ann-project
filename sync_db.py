import os
from src.model_builder import build_model
from src.trainer import save_trials_to_db
import keras_tuner as kt

def force_sync():
    print("--- SYNCING RESULTS TO DATABASE ---")
    
    # 1. Re-initialize the tuner (This loads your existing 0-9 trials)
    # We use the EXACT same settings as your run.py so it finds the files
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, 10000, 46),
        objective='val_accuracy',
        max_trials=10,
        directory='results',
        project_name='reuters_ann'
    )
    
    # 2. Force the tuner to recognize the existing trials
    tuner.reload()
    
    count = len(tuner.oracle.trials)
    print(f"Found {count} trials in the 'results' folder.")
    
    if count > 0:
        # 3. Save to SQL
        save_trials_to_db(tuner)
        print("SUCCESS: Database 'trials.db' has been updated.")
    else:
        print(" ERROR: No trials found. Check your 'results' folder path.")

if __name__ == "__main__":
    force_sync()