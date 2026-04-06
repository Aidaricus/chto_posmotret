# app.py
import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Кэшируем загрузку модели
@st.cache_resource
def load_model():
    return SentenceTransformer('./fine_tuned_movie_search')

# Кэшируем загрузку данных корпуса
@st.cache_data
def load_corpus():
    with open('corpus_descriptions.pkl', 'rb') as f:
        descriptions = pickle.load(f)
    with open('movie_titles.pkl', 'rb') as f:
        titles = pickle.load(f)
    embeddings = np.load('corpus_embeddings.npy')
    return descriptions, titles, embeddings

def search(query, model, corpus_embeddings, corpus_descriptions, corpus_titles, top_k=5):
    # Получаем эмбеддинг запроса
    query_emb = model.encode([query])
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query_emb, corpus_embeddings)[0]
    # Индексы топ-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'title': corpus_titles[idx],
            'description': corpus_descriptions[idx],
            'score': similarities[idx]
        })
    return results

# Интерфейс Streamlit
st.set_page_config(page_title="Поиск фильмов по описанию", layout="wide")
st.title("🍿 Поиск фильмов по запросу")
st.markdown("Введите запрос на русском (например, *фильм про космос* или *хочу романтическую комедию*)")

query = st.text_input("Ваш запрос:", placeholder="например: фильм про любовь и путешествия")

if query:
    with st.spinner("Ищем фильмы..."):
        model = load_model()
        descriptions, titles, embeddings = load_corpus()
        results = search(query, model, embeddings, descriptions, titles, top_k=5)
    
    st.subheader("🎬 Топ-5 релевантных фильмов:")
    for i, res in enumerate(results, 1):
        st.markdown(f"**{i}. {res['title']}**  \n*Схожесть: {res['score']:.2f}*")
        # Обрезаем описание до 300 символов
        short_desc = res['description'][:300] + "..." if len(res['description']) > 300 else res['description']
        st.markdown(f"{short_desc}")
        st.markdown("---")