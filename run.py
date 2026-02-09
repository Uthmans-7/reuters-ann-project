import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.data_loader import load_and_preprocess
from src.trainer import tune_and_train, evaluate_final

def main():
    print("--- STARTING REUTERS AI PROJECT ---")
    
    num_words = 10000
    (X_train, y_train), (X_test, y_test), num_classes = load_and_preprocess(num_words)

    # 1. Run Search & Save ALL Metrics to DB
    best_model, tuner = tune_and_train(
        X_train, y_train, 
        X_test, y_test,   # <--- Added these so we can calc metrics for every trial
        num_words, num_classes,
        max_trials=5,  
        epochs=10       
    )

    # 2. Generate Plot for the Winner
    evaluate_final(best_model, X_test, y_test)
    
    # 3. Save Winner
    best_model.save('reuters_best_model.keras')
    print("\nProcess Complete.")

if __name__ == '__main__':
    main()