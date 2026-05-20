import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. Загрузка и предобработка данных
data0 = pd.read_csv('F:\\1 сем\\МиМИИ\\heart-2.csv')
# print(data0.head())

y = data0['target'].to_numpy().astype(np.float32)
X = data0.drop('target', axis=1).to_numpy().astype(np.float32)

# Нормализация признаков
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Shape of X: {X_scaled.shape}")
print(f"Sample X:\n{X_scaled[:6]}")
print(f"Sample y: {y[:6]}")

# 2. Разделение на train/test
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train shape: {X_train_np.shape}, Test shape: {X_test_np.shape}")

# 3. Преобразование в Tensor PyTorch
# Для BCELoss целевая переменная должна иметь размерность (N, 1) или быть скаляром,
# но удобнее делать (N, 1) для согласованности с выходом модели
X_train_tensor = torch.from_numpy(X_train_np)
y_train_tensor = torch.from_numpy(y_train_np).unsqueeze(1)

X_test_tensor = torch.from_numpy(X_test_np)
y_test_tensor = torch.from_numpy(y_test_np).unsqueeze(1)

# 4. Создание DataLoader
# В исходнике batch_size=1 при обучении
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 5. Определение модели
class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()  # Сигмоида на выходе для вероятности [0, 1]
        )

    def forward(self, x):
        return self.network(x)


input_dim = X_train_np.shape[1]  # Должно быть 13
model = HeartDiseaseModel(input_dim)

# 6. Настройка оптимизатора и функции потерь
# Исходник: loss=binary_crossentropy, optimizer='Adam'
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # LR по умолчанию для Adam

# 7. Цикл обучения
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 8. Оценка модели
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_preds.append(outputs.numpy())
        all_targets.append(targets.numpy())

yy_test_pred_prob = np.concatenate(all_preds).flatten()
yy_test_true = np.concatenate(all_targets).flatten()

# Преобразование вероятностей в классы (0 или 1) для расчета Accuracy
yy_test_pred_classes = (yy_test_pred_prob >= 0.5).astype(int)

# Расчет метрик
# В Keras score[0] - это loss (BCE), score[1] - это accuracy
test_loss = criterion(torch.tensor(yy_test_pred_prob).unsqueeze(1),
                      torch.tensor(yy_test_true).unsqueeze(1)).item()
test_accuracy = accuracy_score(yy_test_true, yy_test_pred_classes)

print(f'Test BinaryCrossentropy (Loss): {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Если нужно просто получить предсказания как в исходнике (вероятности):
# yy_test = yy_test_pred_prob