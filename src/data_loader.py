import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def load_and_preprocess(num_words=10000):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=num_words)
    X_train = vectorize_sequences(X_train, num_words)
    X_test = vectorize_sequences(X_test, num_words)

    num_classes = max(y_train) + 1
    y_train = to_categorical(y_train, num_classes).astype('float32')  # <--- Fix here
    y_test = to_categorical(y_test, num_classes).astype('float32')    # <--- Fix here
    
    return (X_train, y_train), (X_test, y_test), num_classes
