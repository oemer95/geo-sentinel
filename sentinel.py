# Geo-KI: Klassifikation von Landnutzung mit Sentinel-2-Daten

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Dummy-Data
def generate_dummy_data(num_samples=1000, img_size=64, num_classes=4):
    X = np.random.rand(num_samples, img_size, img_size, 3)
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    return X, y

# CNN-Model
def build_model(input_shape=(64, 64, 3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    X, y = generate_dummy_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.summary()

    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.2f}")

    # Beispielbild anzeigen
    plt.imshow(X_test[0])
    plt.title("Test Sample")
    plt.show()

if __name__ == "__main__":
    main()
