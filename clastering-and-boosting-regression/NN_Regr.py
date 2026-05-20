import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import catboost as cb
import warnings

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# Настройка стиля графиков
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# -----------------------------------------------------------------------------
# 1. Загрузка и первичный осмотр данных
# -----------------------------------------------------------------------------
# Предположим, файл называется 'insurance.csv'.
# Если файла нет, создадим демо-датасет из фрагмента, предоставленного в промпте,
# но для полноценной работы лучше использовать полный датасет.
try:
    df = pd.read_csv('F:\\1 сем\\МиМИИ\\insurance.csv')
except FileNotFoundError:
    print("Файл 'insurance.csv' не найден. Создаю пример данных на основе вашего фрагмента.")
    data_sample = """age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
28,male,33,3,no,southeast,4449.462
33,male,22.705,0,no,northwest,21984.47061
32,male,28.88,0,no,northwest,3866.8552
31,female,25.74,0,no,southeast,3756.6216
46,female,33.44,1,no,southeast,8240.5896
37,female,27.74,3,no,northwest,7281.5056
37,male,29.83,2,no,northeast,6406.4107"""
    from io import StringIO
    df = pd.read_csv(StringIO(data_sample))
    print("Используются демонстрационные данные. Для качественной кластеризации и обучения рекомендуется полный датасет.")

print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о типах данных:")
print(df.info())

# -----------------------------------------------------------------------------
# 2. Статистическая обработка и EDA (Exploratory Data Analysis)
# -----------------------------------------------------------------------------
print("\nСтатистическое описание числовых признаков:")
print(df.describe())

print("\nПроверка на пропуски:")
print(df.isnull().sum())

# Графики распределения целевой переменной и ключевых признаков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Распределение charges (целевая переменная)
sns.histplot(df['charges'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribution of Charges')

# 2. Корреляционная матрица (только для числовых признаков)
numeric_cols = ['age', 'bmi', 'children', 'charges']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Correlation Matrix (Numeric)')

# 3. Влияние курения на стоимость
sns.boxplot(x='smoker', y='charges', data=df, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Charges by Smoker Status')

# 4. Влияние региона на стоимость
sns.boxplot(x='region', y='charges', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Charges by Region')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 3. Предобработка и Кластеризация
# -----------------------------------------------------------------------------

# Выделяем признаки для кластеризации (все кроме target)
# Для кластеризации важно масштабировать данные, так как KMeans чувствителен к масштабу
features_for_cluster = ['age', 'bmi', 'children']
# Категориальные признаки тоже можно использовать, если их закодировать,
# но для простоты возьмем численные + закодируем простые бинарные/категориальные для кластеризации

df_prep = df.copy()

# Кодирование категориальных признаков для кластеризации (Label Encoding для простоты расстояний)
df_prep['sex_enc'] = df_prep['sex'].map({'male': 0, 'female': 1})
df_prep['smoker_enc'] = df_prep['smoker'].map({'no': 0, 'yes': 1})
# Region имеет 4 категории, используем простой маппинг или OneHot, но для KMeans лучше непрерывные числа или PCA
# Для упрощения примера возьмем численные признаки + smoker (так как он сильно влияет)
cluster_features = ['age', 'bmi', 'children', 'smoker_enc']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_prep[cluster_features])

# Подбор оптимального количества кластеров (метод локтя)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.xticks(K_range)
plt.show()

# Выбираем K=4 (например, обычно оптимально для таких данных)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_prep['cluster'] = kmeans.fit_predict(scaled_data)

# Визуализация кластеров (на примере age и bmi)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_prep['age'], df_prep['bmi'], c=df_prep['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title(f'Patient Clusters (K={optimal_k})')
plt.show()

# -----------------------------------------------------------------------------
# 4. Подготовка данных для обучения CatBoost
# -----------------------------------------------------------------------------

# Определяем категориальные признаки для CatBoost
cat_features = ['sex', 'smoker', 'region', 'cluster'] # cluster добавляем как категориальный

# Разделяем на признаки и цель
X = df_prep.drop('charges', axis=1)
y = df_prep['charges']

# Убираем вспомогательные закодированные колонки, которые не нужны модели,
# так как оригинальные категориальные уже есть
X = X.drop(columns=['sex_enc', 'smoker_enc'])

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Категориальные признаки: {cat_features}")

# -----------------------------------------------------------------------------
# 5. Обучение модели CatBoost
# -----------------------------------------------------------------------------

model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    cat_features=cat_features,
    verbose=100,
    random_seed=42
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
    plot=True # Вывод графика обучения в notebook/jupyter
)

# -----------------------------------------------------------------------------
# 6. Оценка качества и прогнозирование
# -----------------------------------------------------------------------------

y_pred = model.predict(X_test)

# Расчет метрик
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Результаты модели ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2 Score: {r2:.4f}")

# -----------------------------------------------------------------------------
# 7. Визуализация результатов
# -----------------------------------------------------------------------------

# 1. Сравнение предсказанных и реальных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Medical Charges')
plt.legend()
plt.show()

# 2. Распределение ошибок
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='salmon')
plt.xlabel('Error (Actual - Predicted)')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

# 3. Важность признаков (Feature Importance)
feature_importance = model.get_feature_importance(prettified=True)
feature_importance = feature_importance.sort_values(by='Importances', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importances', y='Feature Id', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Feature Importances (CatBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()