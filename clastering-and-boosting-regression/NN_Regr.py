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

#для чистоты вывода
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

try:
    df = pd.read_csv('F:\\1 сем\\МиМИИ\\insurance.csv')
except FileNotFoundError:
    print("Файл 'insurance.csv' не найден")

print("Первые 5 строк датасета:")
print(df.head())
print("\nИнформация о типах данных:")
print(df.info())

#Статистическая обработка
print("\nСтатистическое описание числовых признаков:")
print(df.describe())

print("\nПроверка на пропуски:")
print(df.isnull().sum())

# Графики распределения целевой переменной и ключевых признаков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#Распределение charges (целевая переменная)
sns.histplot(df['charges'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribution of Charges')

#Корреляционная матрица для числовых признаков
numeric_cols = ['age', 'bmi', 'children', 'charges']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Correlation Matrix (Numeric)')

#Влияние курения на стоимость
sns.boxplot(x='smoker', y='charges', data=df, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Charges by Smoker Status')

#Влияние региона на стоимость
sns.boxplot(x='region', y='charges', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Charges by Region')

plt.tight_layout()
plt.show()

#Предобработка и Кластеризация

#для кластеризации все кроме target
features_for_cluster = ['age', 'bmi', 'children']
df_prep = df.copy()

#Кодирование категориальных признаков
df_prep['sex_enc'] = df_prep['sex'].map({'male': 0, 'female': 1})
df_prep['smoker_enc'] = df_prep['smoker'].map({'no': 0, 'yes': 1})

#возьмем численные признаки + smoker (так как он сильно влияет)
cluster_features = ['age', 'bmi', 'children', 'smoker_enc']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_prep[cluster_features])

#метод локтя
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

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_prep['cluster'] = kmeans.fit_predict(scaled_data)

# Визуализация кластеров на примере age и bmi
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_prep['age'], df_prep['bmi'], c=df_prep['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title(f'Patient Clusters (K={optimal_k})')
plt.show()

#Подготовка данных для обучения CatBoost
cat_features = ['sex', 'smoker', 'region', 'cluster']

X = df_prep.drop('charges', axis=1)
y = df_prep['charges']

X = X.drop(columns=['sex_enc', 'smoker_enc'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер тренировочной выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Категориальные признаки: {cat_features}")

#Обучение модели CatBoost
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
    plot=True
)

#Оценка качества и прогнозирование
y_pred = model.predict(X_test)

#Расчет метрик
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Результаты модели ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2 Score: {r2:.4f}")

#Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Medical Charges')
plt.legend()
plt.show()

#Распределение ошибок
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='salmon')
plt.xlabel('Error (Actual - Predicted)')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

#Важность признаков
feature_importance = model.get_feature_importance(prettified=True)
feature_importance = feature_importance.sort_values(by='Importances', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importances', y='Feature Id', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Feature Importances (CatBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
