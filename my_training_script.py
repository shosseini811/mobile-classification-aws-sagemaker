import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # change this
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # change this
    return model

def input_fn(file_path):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(file_path, target_size=(64, 64), batch_size=32, class_mode='categorical')  # change this

def train_input_fn():
    return input_fn('/opt/ml/input/data/train')

def eval_input_fn():
    return input_fn('/opt/ml/input/data/validation')

def train():
    model = model_fn()
    train_data = train_input_fn()
    eval_data = eval_input_fn()
    history = model.fit(train_data, validation_data=eval_data, epochs=10)

    # Save the training history
    import json
    with open('/opt/ml/model/history.json', 'w') as f:
        json.dump(history.history, f)


if __name__ == "__main__":
    train()
