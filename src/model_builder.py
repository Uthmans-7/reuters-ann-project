from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_model(hp, num_words, num_classes):
    model = Sequential()
    model.add(Input(shape=(num_words,)))

    hp_units1 = hp.Int('units_layer1', 32, 256, step=32)
    model.add(Dense(hp_units1, activation='relu'))

    hp_units2 = hp.Int('units_layer2', 32, 256, step=32)
    model.add(Dense(hp_units2, activation='relu'))

    hp_dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
    if hp_dropout > 0:
        model.add(Dropout(hp_dropout))

    model.add(Dense(int(num_classes), activation='softmax'))

    hp_lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
