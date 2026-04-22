import os
import pickle

import numpy as np
from tensorflow import keras


# Разрешение изображения
IMAGE_RESOLUTION = 192



def load_data(file_path: str) -> np.ndarray:
    """Загружает данные из pickle‑файла."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)



# Загрузка данных для классификации
X_train = load_data('X_train.pickle')
X_test = load_data('X_test.pickle')
y_train_classification = load_data('y_train.pickle')
y_test_classification = load_data('y_test.pickle')

# Загрузка данных для детекции
y_train_detection = load_data('yD_train.pickle')
y_test_detection = load_data('yD_test.pickle')

# Загрузка расписаний изменения скорости обучения
lr_schedule_1 = load_data('lr_schedule1.pickle')
lr_schedule_2 = load_data('lr_schedule2.pickle')


# Модель для классификации
model_classification = keras.models.Sequential([
    keras.layers.InputLayer(shape=(IMAGE_RESOLUTION, IMAGE_RESOLUTION, 1)),
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        use_bias=True
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        use_bias=True
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(
        2,
        activation='sigmoid',
        activity_regularizer=keras.regularizers.L2(1e-6)
    )
])

model_classification.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr_schedule_1),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)
print('Модель для классификации создана.')

# Обучение модели классификации
model_classification.fit(
    X_train,
    keras.utils.to_categorical(y_train_classification),
    batch_size=64,
    epochs=40,
    validation_split=0.2
)
print('Модель для классификации обучена.')

# Оценка модели классификации
loss_classification, accuracy_classification = model_classification.evaluate(
    X_test,
    keras.utils.to_categorical(y_test_classification)
)
print(f'Test Loss (Classification): {loss_classification}')
print(f'Test Accuracy (Classification): {accuracy_classification}')

# Сохранение модели классификации
model_classification.save('model1.keras')
model_classification.save('model1.h5')

# Модель для детекции объекта
model_detection = keras.models.Sequential([
    keras.layers.InputLayer(shape=(IMAGE_RESOLUTION, IMAGE_RESOLUTION, 1)),
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        use_bias=True
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        use_bias=True
    ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(4, activation='relu')
])

model_detection.compile(
    optimizer=keras.optimizers.SGD(learning_rate=lr_schedule_2),
    loss=['mean_absolute_error'],
    metrics=['mean_squared_error']
)
print('Модель для детекции создана.')

# Обучение модели детекции
model_detection.fit(
    X_train,
    keras.utils.to_categorical(y_train_detection),
    batch_size=64,
    epochs=20,
    validation_split=0.2
)
print('Модель для детекции обучена.')

# Оценка модели детекции
loss_detection, mse_detection = model_detection.evaluate(
    X_test,
    keras.utils.to_categorical(y_test_detection)
)
print(f'Test Loss (Detection): {loss_detection}')
print(f'Mean Squared Error (Detection): {mse_detection}')

# Сохранение модели детекции
model_detection.save('model2.keras')
model_detection.save('model2.h5')
