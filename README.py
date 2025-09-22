# Taller-2
# -*- coding: utf-8 -*-
import re
import time
import csv
import emoji
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
stop_words = set(stopwords.words("spanish"))



#  Función de preprocesamiento

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    for word, full in chat_words.items():
        text = re.sub(r"\b" + word + r"\b", full, text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"([!?.,]){2,}", r"\1", text)
    text = re.sub(r"[^a-záéíóúüñ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text, language="spanish")
    tokens = [w for w in tokens if w not in stop_words]
    return tokens



#  Web Scraping de Trustpilot

def get_reviews(url, pages=1):
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



#  MAIN

if __name__ == "__main__":
    url = "https://es.trustpilot.com/review/tradeinn.com"
    reseñas = get_reviews(url, pages=30)  # Cambiá la cantidad de páginas

    save_to_csv(reseñas, r"C:\Users\juani\Documents\dataset.csv")

    # Leer dataset
    df = pd.read_csv(r"C:\Users\juani\Documents\dataset.csv")

    # Contar frecuencias
    all_tokens = " ".join(df["review_procesado"]).split()
    word_freq = Counter(all_tokens)
    print(word_freq.most_common(10))

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
