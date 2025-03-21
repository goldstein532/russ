import os

import numpy as np
import streamlit as st
import pandas as pd
import spacy
import re
from docx import Document
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
from wordcloud import WordCloud
from gensim.models import Word2Vec
import community as community_louvain
import nltk
import streamlit.components.v1 as components
import random
import urllib.parse
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# Загрузка модели spaCy для лемматизации
nlp = spacy.load('ru_core_news_sm')

# Загружаем стоп-слова из NLTK
stop_words = set([
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'ты', 'она',
    'оно', 'мы', 'вы', 'их', 'этот', 'к', 'у', 'за', 'по', 'о', 'мне', 'себя', 'мне', 'было', 'бы', 'когда', 'да', 'но'
])

# Функция для извлечения текста из .docx
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Функция для поиска года в тексте
def extract_year(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return match.group(0) if match else "Не найден"

# Функция для лемматизации текста
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if '-' in token.text and token.is_alpha:
            lemmas.append(token.text.lower())
        elif not token.is_stop and token.is_alpha and token.pos_ not in ['ADP', 'CCONJ', 'PART']:
            lemmas.append(token.lemma_.lower())
    return lemmas

# Функция фильтрации данных по году
def filter_by_year(df, selected_years):
    if "Все годы" in selected_years:
        return df
    return df[df['year'].isin(selected_years)]

# Функция для поиска по леммам с контекстом
def search_by_lemma_with_context(df, search_term, context_size=50):
    results = []
    for _, row in df.iterrows():
        lemmata = lemmatize_text(row['text'])
        if search_term in lemmata:
            # Находим позицию леммы в тексте
            doc = nlp(row['text'])
            for sent in doc.sents:
                if search_term in [token.lemma_.lower() for token in sent]:
                    start = max(0, sent.start_char - context_size)
                    end = min(len(row['text']), sent.end_char + context_size)
                    context = row['text'][start:end].replace(search_term, f'<span style="color: red;">{search_term}</span>')
                    results.append((row['filename'], context))
    return results

# Функция для поиска по буквосочетаниям с контекстом
def search_by_substring_with_context(df, substring, context_size=50):
    results = []
    for _, row in df.iterrows():
        matches = re.finditer(r'\b\w*' + re.escape(substring) + r'\w*\b', row['text'], re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(row['text']), match.end() + context_size)
            context = row['text'][start:end].replace(substring, f'<span style="color: red;">{substring}</span>')
            results.append((row['filename'], context))
    return results

# Функция построения облака слов
def generate_wordcloud(df):
    text = " ".join(df['text'])
    words = lemmatize_text(text)
    filtered_text = " ".join(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Функция для обработки текста
def preprocess_text(text):
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.text.isalpha() and token.text.lower() not in stop_words]
    return words


# Функция для построения интерактивного графа слов
def build_interactive_word_graph_html(df):
    # Преобразование текста в список лемматизированных слов
    sentences = [preprocess_text(text) for text in df['text']]

    # Проверка на наличие данных
    if not sentences or all(len(s) == 0 for s in sentences):
        st.write("Недостаточно данных для построения графа.")
        return

    # Токенизация и очистка текста
    cleaned_sentences = []
    for sentence in sentences:
        cleaned_text = " ".join([word for word in sentence if word not in stop_words])
        cleaned_sentences.append(cleaned_text)

    # Используем TfidfVectorizer для преобразования слов в вектора
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_sentences)  # Преобразуем предложения в вектора

    # Получаем имена всех слов (терминов)
    terms = vectorizer.get_feature_names_out()

    # 1. Вычисление оптимального числа кластеров с использованием метода локтя
    sse = []
    max_clusters = 10  # Максимальное количество кластеров для проверки
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Построение графика для метода локтя
    optimal_clusters = np.argmin(np.diff(sse)) + 2  # Выбираем точку перегиба

    # 2. Применяем KMeans с найденным числом кластеров
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(X)

    # Получаем метки кластеров для слов
    labels = kmeans.labels_

    # Создаем интерактивный граф с PyVis
    net = Network(height='600px', width='800px', notebook=False, bgcolor='#ffffff', font_color='black')

    # Добавляем узлы с цветами, соответствующими кластеру
    for i, word in enumerate(terms):
        cluster_id = labels[i]
        color = f'#{random.randint(0, 0xFFFFFF):06x}'  # Генерация уникального цвета на основе ID кластера
        net.add_node(word, label=word, color=color, title=word,
                     font={'size': 14, 'face': 'arial', 'background': 'white'}, size=20)

    # Рассчитаем частоту совместного появления слов для связей
    co_occurrence = defaultdict(int)
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            word_pair = (sentence[i], sentence[i + 1])
            co_occurrence[word_pair] += 1

    # Добавляем рёбра с учётом частоты совместного появления
    for (word1, word2), count in co_occurrence.items():
        net.add_edge(word1, word2, value=count * 4)  # Умножаем на 4 для нормализации толщины

    # Применяем настройки для улучшения визуализации
    net.force_atlas_2based()  # Использование алгоритма ForceAtlas2 для раскладки узлов
    net.barnes_hut(gravity=-50000, central_gravity=0.3, spring_length=150, spring_strength=0.001, damping=0.09,
                   overlap=1)

    # Сохраняем граф в HTML в файл
    output_path = os.path.join(os.getcwd(), "word_graph.html")
    try:
        net.save_graph(output_path)

        # Генерация HTML-контента для отображения графа в Streamlit
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    background-color: #ffffff;
                }}
                iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <iframe src="file://{os.path.abspath(output_path)}"></iframe>
        </body>
        </html>
        """

        # Вставляем HTML-контент в Streamlit с использованием компонента
        components.html(html_content, height=800)

        # Кнопка для открытия графа в том же окне
        st.markdown("""
        <style>
        .full-screen-btn {
            background-color: #008CBA;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }
        .full-screen-btn:hover {
            background-color: #006F8C;
        }
        </style>
        <button class="full-screen-btn" onclick="document.querySelector('iframe').style.height='100vh';">Развернуть на весь экран</button>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Ошибка при сохранении графа: {e}")

# Функция загрузки файлов
def load_uploaded_files():
    uploaded_files = st.file_uploader("Загрузите документы .docx", type=["docx"], accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            text = extract_text_from_docx(uploaded_file)
            year = extract_year(text)
            documents.append({'filename': uploaded_file.name, 'year': year, 'text': text})
        return pd.DataFrame(documents)
    return pd.DataFrame()

# Интерфейс Streamlit
st.title('Лингвистический корпус: Документы стратегического планирования РФ')
df = load_uploaded_files()

if not df.empty:
    # Убираем строки, где год не найден
    df = df[df['year'] != "Не найден"]

    # Получаем список уникальных годов и сортируем их
    years = ["Все годы"] + sorted(df['year'].unique().tolist())

    # Выбор года для фильтрации
    selected_years = st.sidebar.multiselect('Выберите год', years, default="Все годы")

    # Фильтрация данных по выбранному году
    filtered_df = filter_by_year(df, selected_years)

    # Отображение выбранных документов
    st.subheader("Выбранные документы")
    if not filtered_df.empty:
        st.write(filtered_df[['filename', 'year']])
    else:
        st.write("Нет документов, соответствующих выбранным годам.")

    menu = [
        'Поиск по леммам', 'Поиск по буквосочетаниям', 'Частотный словарь',
        'Коллокации', 'Конкорданс', 'Облако слов', 'Граф связей слов'
    ]
    choice = st.sidebar.radio('Выберите функцию', menu)

    if choice == 'Поиск по леммам':
        search_term = st.text_input('Введите лемму')
        context_size = st.slider('Размер контекста', 10, 100, 50)
        if search_term:
            results = search_by_lemma_with_context(filtered_df, search_term, context_size)
            for filename, context in results:
                st.write(f"**Файл:** {filename}")
                st.write(f"**Контекст:** {context}", unsafe_allow_html=True)

    elif choice == 'Поиск по буквосочетаниям':
        substring = st.text_input('Введите буквосочетание')
        context_size = st.slider('Размер контекста', 10, 100, 50)
        if substring:
            results = search_by_substring_with_context(filtered_df, substring, context_size)
            for filename, context in results:
                st.write(f"**Файл:** {filename}")
                st.write(f"**Контекст:** {context}", unsafe_allow_html=True)

    elif choice == 'Частотный словарь':
        freq_dict = Counter(lemmatize_text(" ".join(filtered_df['text'])))
        st.write(pd.DataFrame(freq_dict.items(), columns=["Слово", "Частота"]))

    elif choice == 'Облако слов':
        generate_wordcloud(filtered_df)

    elif choice == 'Граф связей слов':
        build_interactive_word_graph_html(filtered_df)

else:
    st.write("Пожалуйста, загрузите файлы.")
