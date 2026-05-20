import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# --- 1. КОНФИГУРАЦИЯ ПУТЕЙ И ПАРАМЕТРОВ ---
TRAIN_CSV_PATH = "C:\\Users\\swafferinian\\Downloads\\rusentiment_preselected_posts.csv"
TEST_CSV_PATH = "C:\\Users\\swafferinian\\Downloads\\rusentiment_test.csv"

MODEL_NAME = "DeepPavlov/rubert-base-cased"
OUTPUT_DIR = "./rubert-local-csv-finetuned"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Маппинг меток для RuSentiment
# В оригинальном датасете часто: negative, neutral, positive, skip, speech
# Мы приводим их к: 0 (Neg), 1 (Pos), 2 (Neu)
LABEL_MAP = {0: "Negative", 1: "Positive", 2: "Neutral"}
ALLOWED_LABELS = {0, 1, 2}

# --- 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ИЗ CSV ---
print("Загрузка CSV файлов...")


def load_and_clean_csv(file_path):
    """
    Загружает CSV, нормализует метки и фильтрует данные.
    """
    # Читаем CSV. sep=None позволяет pandas автоматически определить разделитель (запятая или точка с запятой)
    df = pd.read_csv(file_path, sep=None, engine='python')

    # Проверяем наличие нужных колонок
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"В файле {file_path} не найдены колонки 'label' и 'text'. Текущие: {df.columns.tolist()}")

    # Оставляем только нужные колонки
    df = df[['label', 'text']].copy()

    # Удаляем строки с NaN
    df.dropna(subset=['text', 'label'], inplace=True)

    # Приводим label к строке для унификации обработки
    df['label'] = df['label'].astype(str).str.strip().str.lower()

    # Маппинг текстовых меток в числовые
    # negative -> 0, positive -> 1, neutral -> 2
    mapping_dict = {
        'negative': 0,
        'positive': 1,
        'neutral': 2,
        'neg': 0,
        'pos': 1,
        'neu': 2
    }

    # Применяем маппинг. Если метки уже были числами (0, 1, 2), они станут строками '0', '1', '2'
    # и не найдутся в словаре, поэтому добавим обработку для них
    def convert_label(lbl):
        if lbl in mapping_dict:
            return mapping_dict[lbl]
        try:
            # Пробуем привести к int, если это было число
            val = int(float(lbl))
            if val in ALLOWED_LABELS:
                return val
        except:
            pass
        return -1  # Неподходящая метка

    df['label'] = df['label'].apply(convert_label)

    # Фильтруем по допустимым меткам (убираем -1, т.е. skip, speech и прочее)
    df = df[df['label'].isin(ALLOWED_LABELS)]

    # Удаляем пустые тексты
    df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

    # Приводим label обратно к int
    df['label'] = df['label'].astype(int)

    return df


try:
    train_df = load_and_clean_csv(TRAIN_CSV_PATH)
    test_df = load_and_clean_csv(TEST_CSV_PATH)
except Exception as e:
    print(f"Ошибка при загрузке CSV: {e}")
    exit()

print(f"Train samples after cleaning: {len(train_df)}")
print(f"Test samples after cleaning: {len(test_df)}")
print(f"Unique labels in Train: {train_df['label'].unique()}")

# Преобразуем Pandas DataFrame в Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# --- 3. ТОКЕНИЗАЦИЯ ---
print("Инициализация токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


# Применяем токенизацию
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# --- 4. МОДЕЛЬ И МЕТРИКИ ---
print("Загрузка модели...")
num_labels = len(ALLOWED_LABELS)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


# --- 5. ОБУЧЕНИЕ ---
# ИСПРАВЛЕНИЕ: evaluation_strategy заменено на eval_strategy
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  # <--- ИСПРАВЛЕНО
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Начало обучения...")
trainer.train()

# Сохранение модели
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")


# --- 6. ГРАФИКИ ОБУЧЕНИЯ ---
def plot_training_history(trainer):
    log_history = trainer.state.log_history
    # Фильтруем логи, оставляя только те, где есть результаты оценки (конец эпохи)
    history_df = [log for log in log_history if 'eval_loss' in log]

    if not history_df:
        print("Нет данных для построения графиков.")
        return

    epochs = [log['epoch'] for log in history_df]
    eval_losses = [log['eval_loss'] for log in history_df]
    eval_accs = [log['eval_accuracy'] for log in history_df]
    eval_f1s = [log['eval_f1'] for log in history_df]

    # График Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, eval_losses, label='Eval Loss', marker='o', color='red')
    plt.title('Evaluation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('loss_plot.png')
    plt.show()

    # График Метрик
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, eval_accs, label='Accuracy', marker='s', color='blue')
    plt.plot(epochs, eval_f1s, label='F1 Score', marker='^', color='green')
    plt.title('Evaluation Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('metrics_plot.png')
    plt.show()


plot_training_history(trainer)


# --- 7. ТЕСТИРОВАНИЕ И ОТЧЕТЫ ---
def test_model_on_samples(model, tokenizer, dataset, num_samples=5):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Выбираем случайные индексы
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print("\n--- ПРИМЕРЫ ПРЕДСКАЗАНИЙ (Случайные из теста) ---")
    for i in indices:
        item = dataset[i]
        text = item['text']
        true_label = item['label']

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        print(f"Текст: \"{str(text)[:80]}...\"")
        print(
            f"Истина: {LABEL_MAP.get(true_label, true_label)} | Предсказание: {LABEL_MAP.get(predicted_class, predicted_class)} (Уверенность: {confidence:.2f})")
        print("-" * 60)


# Запуск тестов на примерах
test_model_on_samples(trainer.model, tokenizer, test_dataset, num_samples=5)

# Полный классификационный отчет
print("\n--- CLASSIFICATION REPORT ---")
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

target_names = [LABEL_MAP[i] for i in sorted(ALLOWED_LABELS)]
print(classification_report(labels, preds, target_names=target_names))
