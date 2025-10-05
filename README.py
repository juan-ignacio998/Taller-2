# Taller-2
# -*- coding: utf-8 -*-
import re
import time
import csv
import importlib.util
from collections import Counter
if importlib.util.find_spec("nltk") is not None:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
else:
    stopwords = None
    word_tokenize = None


if importlib.util.find_spec("emoji") is not None:
    import emoji  # type: ignore
else:
    emoji = None


#  Diccionario de chat words
chat_words = {
    "q": "que",
    "xq": "porque",
    "k": "que",
    "tmb": "también",
    "d": "de",
    "m": "me",
}

# Stopwords en español


def load_spanish_stopwords():
    """Carga las stopwords en español con un conjunto mínimo de respaldo."""

    if stopwords is None:
        return {
            "de",
            "la",
            "que",
            "el",
            "en",
            "y",
            "a",
            "los",
            "del",
            "se",
        }

    try:
        return set(stopwords.words("spanish"))
    except LookupError:
        # Conjunto mínimo para escenarios sin corpus descargado.
        return {
            "de",
            "la",
            "que",
            "el",
            "en",
            "y",
            "a",
            "los",
            "del",
            "se",
        }


stop_words = load_spanish_stopwords()



#  Función de preprocesamiento

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    for word, full in chat_words.items():
        text = re.sub(r"\b" + word + r"\b", full, text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"([!?.,]){2,}", r"\1", text)
    text = re.sub(r"[^a-záéíóúüñ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if word_tokenize is None:
        tokens = text.split()
    else:
        try:
            tokens = word_tokenize(text, language="spanish")
        except LookupError:
            tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens



#  Web Scraping de Trustpilot

def get_reviews(url, pages=1):
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.93 Safari/537.36"
        )
    }

    all_reviews = []

    for page in range(1, pages + 1):
        print(f"Scrapeando página {page}...")
        paged_url = f"{url}?page={page}"
        response = requests.get(paged_url, headers=headers)

        if response.status_code != 200:
            print(f"Error {response.status_code} en {paged_url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        reviews = soup.find_all("p", {"data-service-review-text-typography": "true"})

        for r in reviews:
            all_reviews.append(r.get_text().strip())

        time.sleep(2)  # respetar servidor

    return all_reviews



#  Guardar en CSV

def save_to_csv(data, filename="dataset.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_original", "review_procesado"])
        for review in data:
            tokens = preprocess_text(review)
            writer.writerow([review, " ".join(tokens)])
    print(f"✅ CSV guardado en {filename} con {len(data)} reseñas.")



#  Análisis de sentimientos

_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Carga perezosamente un modelo de análisis de sentimientos en español."""

    global _sentiment_analyzer

    if _sentiment_analyzer is None:
        from transformers import pipeline

        _sentiment_analyzer = pipeline(
            "sentiment-analysis", model="finiteautomata/beto-sentiment-analysis"
        )

    return _sentiment_analyzer


def analyze_sentiment(reviews, analyzer=None):
    """Devuelve etiquetas y puntajes de sentimiento para una lista de reseñas."""

    if analyzer is None:
        analyzer = get_sentiment_analyzer()
    results = analyzer(reviews, truncation=True)
    labels = [result["label"].lower() for result in results]
    scores = [result["score"] for result in results]

    return labels, scores


#  MAIN

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    url = "https://es.trustpilot.com/review/tradeinn.com"
    dataset_path = "dataset.csv"

    reseñas = get_reviews(url, pages=30)  # Cambiá la cantidad de páginas

    save_to_csv(reseñas, dataset_path)

    # Leer dataset
    df = pd.read_csv(dataset_path)

    # Análisis de sentimientos
    sentimientos, puntajes = analyze_sentiment(df["review_original"].tolist())
    df["sentimiento"] = sentimientos
    df["puntaje_sentimiento"] = puntajes
    df.to_csv(dataset_path, index=False)

    # Contar frecuencias
    all_tokens = " ".join(df["review_procesado"]).split()
    word_freq = Counter(all_tokens)
    print(word_freq.most_common(10))

    # Distribución de sentimientos
    print("Distribución de sentimientos:")
    print(Counter(sentimientos))

    # Nube de palabras
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(" ".join(all_tokens))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
