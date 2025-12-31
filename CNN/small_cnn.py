import numpy as np
import os, random
import tensorflow as tf
from tensorflow import keras
from keras import layers

(train_X, train_y), (test_X, test_y) = keras.datasets.cifar10.load_data()

train_y = train_y.squeeze().astype(np.int64)
test_y = test_y.squeeze().astype(np.int64)
# normalization
train_X = train_X.astype(np.float32) / 255.0
test_X = test_X.astype(np.float32) / 255.0

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# This is my second version of cnn model, which is significantly smaller then the first one
def build_model(use_dropout=False, use_batchnorm=False, dropout_rate=0.3):
    inputs = keras.Input(shape=(32,32,3))
    x = inputs

    
    x = layers.Conv2D(16,(3,3),padding="same",activation="relu")(x)
    x = layers.Conv2D(16,(3,3),padding="same",activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    if use_dropout: x = layers.Dropout(dropout_rate)(x)

    
    x = layers.Conv2D(32,(3,3),padding="same",activation="relu")(x)
    x = layers.Conv2D(32,(3,3),padding="same",activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    if use_dropout: x = layers.Dropout(dropout_rate)(x)

    
    x = layers.Conv2D(64,(3,3),padding="same",activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    if use_dropout: x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

def train_and_eval(model, name, train_X, train_y, test_X, test_y, epochs=20, batch_size=64):
    compile_model(model)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    print(f"\n=== {name} ===")
    print("Params:", model.count_params())
    history = model.fit(
        train_X, train_y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
    print(f"{name} | Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # The best version
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])
    return {
        "name": name,
        "params": model.count_params(),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }

# different models
models = [
    ("Baseline", build_model(use_dropout=False, use_batchnorm=False)),
    ("+ Dropout(0.3)", build_model(use_dropout=True, use_batchnorm=False, dropout_rate=0.3)),
    ("+ BatchNorm", build_model(use_dropout=False, use_batchnorm=True)),
    ("+ BatchNorm + Dropout(0.3)", build_model(use_dropout=True, use_batchnorm=True, dropout_rate=0.3)),
]

results = []
for name, m in models:
    results.append(train_and_eval(m, name, train_X, train_y, test_X, test_y, epochs=15, batch_size=32))

print("\n=== Summary ===")
for r in results:
    print(
        f"{r['name']:<28} params={r['params']:<9} "
        f"best_val_acc={r['best_val_acc']:.4f} test_acc={r['test_acc']:.4f}"
    )

# Used google colab to generate results:

# === Summary ===
# Baseline                         best_val_acc=0.7620 test_acc=0.7449
# + Dropout(0.3)                   best_val_acc=0.7360 test_acc=0.7297
# + BatchNorm                      best_val_acc=0.7580 test_acc=0.7493
# + BatchNorm + Dropout(0.3)       best_val_acc=0.7634 test_acc=0.7476

# each model had params=101402
# and it seems like additional BatchNorm has only a very little positive influence on the test accuracy
# meanwhile Dropout has caused worse test and best_val accuracy

# Compering to my first model, this one produces way worse outcomes and I am now sure that bigger model (in this case) produces better outcomes