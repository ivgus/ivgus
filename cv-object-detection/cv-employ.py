import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras



def load_and_convert_data(file_path: str) -> np.ndarray:
    """Загружает данные из pickle‑файла и преобразует в numpy array типа float32."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return np.array(data, dtype=np.float32)



# --- Загрузка данных ---
X_train = load_and_convert_data('X_train.pickle')
y_train = load_and_convert_data('y_train.pickle')
X_test = load_and_convert_data('X_test.pickle')
y_test = load_and_convert_data('y_test.pickle')

# --- Проверка размеров и обрезка ---
min_len = min(len(X_test), len(y_test))
X_test = X_test[:min_len]
y_test = y_test[:min_len]

print('Before reshape:')
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# --- Загрузка модели ---
model = keras.models.load_model('model1.keras')

print(f'Model input shape: {model.input_shape}')
print(f'Model output shape: {model.output_shape}')

# --- Приведение формы меток под выход модели ---
output_dim = model.output_shape[1]  # Должно быть 2

if y_train.ndim == 1:
    y_train = np.stack([y_train] * output_dim, axis=1)
if y_test.ndim == 1:
    y_test = np.stack([y_test] * output_dim, axis=1)

print('After reshape:')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# --- Настройка оптимизатора с экспоненциальным снижением learning rate ---
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=130,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mean_absolute_error',
    metrics=[keras.metrics.RootMeanSquaredError()]
)

# --- Обучение модели ---
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# --- Визуализация истории обучения ---
def plot_history(training_history: keras.callbacks.History) -> None:
    """Строит графики потерь (MAE) и метрики RMSE в процессе обучения."""
    plt.figure(figsize=(12, 5))

    # График MAE
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['loss'], label='Train Loss (MAE)')
    plt.plot(training_history.history['val_loss'], label='Val Loss (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MAE over Epochs')
    plt.legend()

    # График RMSE
    plt.subplot(1, 2, 2)
    plt.plot(
        training_history.history['root_mean_squared_error'],
        label='Train RMSE'
    )
    plt.plot(
        training_history.history['val_root_mean_squared_error'],
        label='Val RMSE'
    )
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()



# --- Дополнительные графики анализа ---
def plot_detailed_analysis(
    training_history: keras.callbacks.History,
    trained_model: keras.Model,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    learning_rate_schedule: keras.optimizers.schedules.LearningRateSchedule
) -> None:
    """
    Строит расширенные графики: ошибки, learning rate, предсказания vs истинные значения,
    а также гистограмму распределения абсолютной ошибки.
    """
    # Графики MAE, RMSE и learning rate
    plt.figure(figsize=(14, 5))

    # MAE и RMSE
    plt.subplot(1, 3, 1)
    plt.plot(training_history.history['loss'], label='Train MAE')
    plt.plot(training_history.history['val_loss'], label='Val MAE')
    plt.plot(
        training_history.history['root_mean_squared_error'],
        label='Train RMSE'
    )
    plt.plot(
        training_history.history['val_root_mean_squared_error'],
        label='Val RMSE'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Losses (MAE / RMSE)')
    plt.legend()

    # Learning rate по эпохам
    lrs = [
        learning_rate_schedule(epoch).numpy()
        for epoch in range(len(training_history.history['loss']))
    ]
    plt.subplot(1, 3, 2)
    plt.plot(lrs, label='Learning Rate', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate Decay')
    plt.legend()

    # Предсказания vs истинные значения
    y_pred = trained_model.predict(test_features)
    plt.subplot(1, 3, 3)
    plt.scatter(test_labels[:, 0], y_pred[:, 0], alpha=0.5, label='X coord')
    plt.scatter(test_labels[:, 1], y_pred[:, 1], alpha=0.5, label='Y coord')
    plt.plot(
        [test_labels.min(), test_labels.max()],
        [test_labels.min(), test_labels.max()],
        'k--',
        lw=1
    )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Гистограмма абсолютной ошибки
    errors = np.abs(y_pred - test_labels)
    abs_errors = np.linalg.norm(errors, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(abs_errors, bins=30, color='tomato', edgecolor='black')
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('|| Prediction - Ground Truth ||')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# --- Построение графиков ---
plot_history(history)
plot_detailed_analysis(history, model, X_test, y_test, lr_schedule)

# --- Оценка модели на тестовых данных ---
test_loss, test_rmse = model.evaluate(X_test, y_test)
print("\nFinal Evaluation:")
print(f'Test MAE: {test_loss:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')
