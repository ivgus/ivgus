import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize, WhitespaceTokenizer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy, binary_crossentropy
from keras.utils import plot_model
import itertools
import graphviz

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Константы
DATA_FILE_PATH = r'C:\Users\swafferinian\Desktop\data.txt'
TEST_FILE_PATH = r'C:\Users\swaffernian\Desktop\t.txt'
MODEL_SAVE_PATH = 'word2vec.model'
MODEL1_PLOT_PATH = 'model1.png'
MODEL2_PLOT_PATH = 'model2.png'
VECTOR_SIZE = 100
WINDOW_SIZE = 4
MIN_COUNT = 1
WORKERS_COUNT = 4
SKIP_GRAM = 1  # sg=0 CBOW; sg=1 Skip-Gram
EPOCHS_MODEL1 = 40
EPOCHS_MODEL2 = 10
BATCH_SIZE_MODEL1 = 1
BATCH_SIZE_EVALUATION = 50
BATCH_SIZE_MODEL2 = 10
TRAIN_SIZE_MODEL1 = 300
TRAIN_SIZE_MODEL2 = 1000
MAX_WORDS_FOR_VISUALIZATION = 50

# Инициализация инструментов обработки текста
tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
lancaster = LancasterStemmer()



def load_and_preprocess_data(file_path):
    """Загрузка и предобработка данных из файла."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    for i in range(len(data)):
        data[i] = data[i].replace(",", " ").replace("\n", "")
        data[i] = tokenizer.tokenize(data[i])
    return data



def create_word2vec_model(sentences, vector_size, window, min_count, workers, sg):
    """Создание и сохранение модели Word2Vec."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg
    )
    model.save(MODEL_SAVE_PATH)
    return model



def prepare_labels_and_texts(data_raw):
    """Подготовка меток и текстов для обучения."""
    d0 = [0] * 3
    d1 = [d0 for _ in data_raw]
    data_processed = []

    for i, line in enumerate(data_raw):
        if line[0] == "И":
            d1[i] = [1, 0, 0]
        elif line[0] == "Т":
            d1[i] = [0, 1, 0]
        elif line[0] == "П":
            d1[i] = [0, 0, 1]

        # Очистка текста
        line = (
            line.replace("Имя", "")
            .replace("Телефонный ", "")
            .replace("Телефон", "")
            .replace("Почта", "")
            .replace(",", "")
            .replace("\n", "")
        )
        line_tokens = tokenizer.tokenize(line)
        data_processed.append(line_tokens)

    return d1, data_processed



def calculate_average_vectors(data_tokens, word2vec_model):
    """Вычисление средних векторов для предложений."""
    data_vectors = []
    for tokens in data_tokens:
        if tokens:  # Проверка на пустые списки
            vector_sum = sum(word2vec_model.wv[token] for token in tokens)
            average_vector = vector_sum / len(tokens)
            data_vectors.append(average_vector)
        else:
            data_vectors.append(np.zeros(word2vec_model.vector_size))
    return np.array(data_vectors)



def build_and_train_model1(input_dim, x_train, y_train, epochs, batch_size):
    """Построение и обучение первой модели нейронной сети."""
    model1 = Sequential([
        Dense(200, activation='relu', input_dim=input_dim),
        Dense(300, activation='relu'),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax')
    ])

    plot_model(model1, to_file=MODEL1_PLOT_PATH)
    model1.compile(loss=CategoricalCrossentropy(), optimizer='Adam', metrics=['accuracy'])
    history1 = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model1, history1



def build_and_train_model2(input_dim, x_train, y_train, epochs, batch_size):
    """Построение и обучение второй модели нейронной сети."""
    model2 = Sequential([
        Dense(200, activation='relu', input_dim=input_dim),
        Dense(300, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    plot_model(model2, to_file=MODEL2_PLOT_PATH)
    model2.compile(loss=binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    history2 = model2.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model2, history2



def plot_training_history(history, model_name):
    """Визуализация истории обучения модели."""
    plt.figure(figsize=(12, 5))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title(f'{model_name} accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title(f'{model_name} loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.show()



def visualize_word_embeddings(word2vec_model, max_words=50):
    """Визуализация векторных представлений слов с помощью PCA."""
    word_vectors = word2vec_model.wv[word2vec_model.wv.index_to_key]
    pca = PCA(n_components=2)
    result = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(result[:max_words, 0], result[:max_words, 1])

    words = list(word2vec_model.wv.index_to_key)[:max_words]
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=12)

    plt.title("Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()


def interactive_prediction(word2vec_model, model1, model2):
    """Интерактивное предсказание на основе пользовательского ввода."""
    while True:
        print("Введите запрос")
        query = input().replace("\n", "").replace(",", " ")
        query_tokens = tokenizer.tokenize(query)

        # Обновление модели Word2Vec с учётом нового запроса
        word2vec_model.build_vocab([query_tokens], update=True)
        word2vec_model.train([query_tokens], epochs=word2vec_model.epochs, total_examples=word2vec_model.corpus_count)

        # Вычисление векторов для запроса
        if query_tokens:
            query_vectors = [word2vec_model.wv[token] for token in query_tokens if token in word2vec_model.wv]
            if query_vectors:
                # Средний вектор для всего запроса
                query_vector_avg = np.mean(query_vectors, axis=0).reshape(1, -1)
                # Векторы для каждого слова
                query_vectors_array = np.array(query_vectors)
            else:
                print("Ни одно из слов запроса не найдено в модели Word2Vec.")
                continue
        else:
            print("Пустой запрос.")
            continue

        # Предсказание первой модели (классификация по трём классам)
        prediction1 = model1.predict(query_vector_avg)
        print(f"Предсказание модели 1 (вероятности классов): {prediction1[0]}")

        # Предсказание второй модели (бинарная классификация)
        prediction2 = model2.predict(query_vectors_array)
        print(f"Предсказание модели 2 (вероятности): {prediction2.flatten()}")

        # Определение наиболее вероятного класса из первой модели
        predicted_class_idx = np.argmax(prediction1[0])
        predicted_word_idx = np.argmax(prediction2)

        # Получение слова из запроса, соответствующего максимальной вероятности во второй модели
        if predicted_word_idx < len(query_tokens):
            predicted_word = query_tokens[predicted_word_idx]
        else:
            predicted_word = query_tokens[-1]  # запасной вариант

        # Корректировка окончания слова в зависимости от класса
        if np.max(prediction1[0]) == prediction1[0][0]:  # Класс 0
            if predicted_word.endswith('ея') or predicted_word.endswith('ия'):
                predicted_word = predicted_word[:-1] + 'й'
            elif predicted_word.endswith('а') or predicted_word.endswith('е'):
                predicted_word = predicted_word[:-1]

        print(f"Предсказанное слово: {predicted_word}")

        # Запрос на завершение работы
        finish_input = input("Закончить? 1 — да, 0 — нет: ")
        if finish_input.strip() == '1':
            break



# Основной блок выполнения программы
if __name__ == "__main__":
    # Загрузка и предобработка данных
    data = load_and_preprocess_data(DATA_FILE_PATH)

    # Создание модели Word2Vec
    word2vec_model = create_word2vec_model(
        data, VECTOR_SIZE, WINDOW_SIZE, MIN_COUNT, WORKERS_COUNT, SKIP_GRAM
    )

    # Поиск похожих слов
    similar_words = word2vec_model.wv.most_similar(positive=["Артура"])
    print(f"Похожие слова на 'Артура': {similar_words}")

    # Подготовка меток и текстов
    labels, processed_texts = prepare_labels_and_texts(data)

    # Вычисление средних векторов
    average_vectors = calculate_average_vectors(processed_texts, word2vec_model)
    labels_array = np.array(labels)

    # Разделение на обучающую и тестовую выборки для модели 1
    X_train_model1 = average_vectors[:TRAIN_SIZE_MODEL1]
    y_train_model1 = labels_array[:TRAIN_SIZE_MODEL1]
    X_test_model1 = average_vectors[TRAIN_SIZE_MODEL1:]
    y_test_model1 = labels_array[TRAIN_SIZE_MODEL1:]

    # Обучение первой модели
    model1, history1 = build_and_train_model1(
        VECTOR_SIZE, X_train_model1, y_train_model1, EPOCHS_MODEL1, BATCH_SIZE_MODEL1
    )

    # Оценка первой модели
    score1 = model1.evaluate(X_test_model1, y_test_model1, batch_size=BATCH_SIZE_EVALUATION)
    print(f'Test CategoricalCrossentropy: {score1[0]}')
    print(f'Test accuracy: {score1[1]}')

    # Визуализация истории обучения модели 1
    plot_training_history(history1, "Model1")

    # Загрузка дополнительных данных для второй модели
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as file:
        test_data = file.readlines()

    # Обработка данных для второй модели
    flat_texts = list(itertools.chain(*processed_texts))
    word_vectors_list = [word2vec_model.wv[word] for word in flat_texts if word in word2vec_model.wv]
    additional_labels = [float(line.replace("\n", "")) for line in test_data]

    X_model2 = np.array(word_vectors_list)
    y_model2 = np.array(additional_labels)

    # Разделение на выборки для модели 2
    X_train_model2 = X_model2[:TRAIN_SIZE_MODEL2]
    y_train_model2 = y_model2[:TRAIN_SIZE_MODEL2]
    X_test_model2 = X_model2[TRAIN_SIZE_MODEL2:]
    y_test_model2 = y_model2[TRAIN_SIZE_MODEL2:]

    # Обучение второй модели
    model2, history2 = build_and_train_model2(
        VECTOR_SIZE, X_train_model2, y_train_model2, EPOCHS_MODEL2, BATCH_SIZE_MODEL2
    )

    # Оценка второй модели
    score2 = model2.evaluate(X_test_model2, y_test_model2, batch_size=BATCH_SIZE_EVALUATION)
    print(f'Test BinaryCrossentropy: {score2[0]}')
    print(f'Test accuracy: {score2[1]}')

    # Визуализация истории обучения модели 2
    plot_training_history(history2, "Model2")

    # Визуализация векторных представлений слов
    visualize_word_embeddings(word2vec_model, MAX_WORDS_FOR_VISUALIZATION)

    # Запуск интерактивного режима предсказания
    interactive_prediction(word2vec_model, model1, model2)
