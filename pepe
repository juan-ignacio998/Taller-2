import os
import re
import time
import emoji
import nltk
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from wordcloud import WordCloud

# ============================
# Configuración general
# ============================

URL = "https://es.trustpilot.com/review/tradeinn.com"
PAGINAS = 100        # número de páginas a scrapear
DELAY = 2.0         # segundos entre solicitudes
ARCHIVO_SALIDA = "dataset_labeled.csv"
OMITIR_SCRAPEO = False  # ponlo en True si ya tienes el CSV

# Pesos de clase y configuración de reporte
PESOS_CLASE = {"neg": 1.0, "neu": 6.0, "pos": 1.0}  # antes 3.0
ZERO_DIV = 0  # evita warnings si alguna clase no recibe predicciones

# Descargar recursos de NLTK
for pkg in ["stopwords", "punkt"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

STOP_WORDS = set(stopwords.words("spanish"))

CHAT_WORDS = {"q": "que", "xq": "porque", "k": "que", "tmb": "también", "d": "de", "m": "me"}

# ============================
# Preprocesamiento de texto
# ============================

def limpiar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = emoji.demojize(texto, delimiters=(" ", " "))
    for corto, completo in CHAT_WORDS.items():
        texto = re.sub(rf"\b{re.escape(corto)}\b", completo, texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"([!?.,]){2,}", r"\1", texto)
    texto = re.sub(r"[^a-záéíóúüñ ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    tokens = word_tokenize(texto, language="spanish")
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

# ============================
# Scraping de Trustpilot
# ============================

CABECERAS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36"}

def obtener_rating_desde_alt(alt_text):
    if not alt_text:
        return None
    alt_text = alt_text.lower()
    if "estrella" not in alt_text and "star" not in alt_text:
        return None
    m = re.search(r"(\d)\s*(?:de|out of)?\s*5", alt_text)
    return int(m.group(1)) if m else None

def extraer_reseñas(soup: BeautifulSoup):
    reseñas = []
    bloques = soup.find_all("section") or soup.find_all("article")
    if not bloques:
        bloques = soup.find_all("p", {"data-service-review-text-typography": "true"})
    for b in bloques:
        p = b.find("p", {"data-service-review-text-typography": "true"}) or b.find("p")
        texto = p.get_text(strip=True) if p else None
        rating = None
        tag = b.find(attrs={"data-service-review-rating": True})
        if tag and tag.get("data-service-review-rating"):
            try:
                rating = int(tag["data-service-review-rating"])
            except:
                rating = None
        if rating is None:
            img = b.find("img", alt=True)
            if img:
                rating = obtener_rating_desde_alt(img.get("alt"))
        if texto and rating:
            reseñas.append((texto, rating))
    return reseñas

def obtener_reseñas(url: str, paginas: int = 1, delay: float = 2.0):
    filas = []
    for pagina in range(1, paginas + 1):
        print(f"Scrapeando página {pagina} de {paginas}")
        resp = requests.get(f"{url}?page={pagina}", headers=CABECERAS, timeout=25)
        if resp.status_code != 200:
            print(f"Error {resp.status_code} en la página {pagina}")
            time.sleep(delay)
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        pares = extraer_reseñas(soup)
        for texto, rating in pares:
            filas.append({
                "review_original": texto,
                "review_clean": limpiar_texto(texto),
                "Rating": rating
            })
        time.sleep(delay)
    return pd.DataFrame(filas)

# ============================
# Etiquetado (1–5 → neg/neu/pos)
# ============================

def etiquetar_sentimiento(r):
    if r <= 2:
        return "neg"
    elif r == 3:
        return "neu"
    else:
        return "pos"

# ============================
# Entrenamiento y evaluación
# ============================

def entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test, nombre):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"=== {nombre} ===")
    print(classification_report(y_test, y_pred, digits=3, zero_division=ZERO_DIV))
    return f1_score(y_test, y_pred, average="macro")

# ============================
# Main
# ============================
if __name__ == "__main__":
    if OMITIR_SCRAPEO and os.path.exists(ARCHIVO_SALIDA):
        df = pd.read_csv(ARCHIVO_SALIDA)
    else:
        df = obtener_reseñas(URL, paginas=PAGINAS, delay=DELAY)
        df = df.dropna(subset=["review_clean", "Rating"]).drop_duplicates(subset=["review_original"])
        df["Sentiment"] = df["Rating"].apply(etiquetar_sentimiento)
        df.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8")

    print("\nDistribución de Rating (1–5):")
    print(df["Rating"].value_counts().sort_index())
    print("\nDistribución de Sentiment (neg/neu/pos):")
    print(df["Sentiment"].value_counts())

    # Nube de palabras
    tokens = " ".join(df["review_clean"].astype(str))
    wc = WordCloud(width=1000, height=500, background_color="white").generate(tokens)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras — corpus limpio")
    plt.show()

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_clean"].astype(str), df["Sentiment"].astype(str),
        test_size=0.2, random_state=42, stratify=df["Sentiment"]
    )

    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Baseline
    clase_mayoritaria = Counter(y_train).most_common(1)[0][0]
    y_pred_base = [clase_mayoritaria] * len(y_test)
    print("\n=== Baseline (clase mayoritaria) ===")
    print(classification_report(y_test, y_pred_base, digits=3, zero_division=ZERO_DIV))

    # Modelos
    resultados = []
    modelos = [
        ("Naïve Bayes (TF-IDF)", MultinomialNB(alpha=0.5)),
        ("Regresión Logística (TF-IDF)", LogisticRegression(max_iter=2000, class_weight=PESOS_CLASE, solver="lbfgs", C=1.0)),
        ("SVM Lineal (TF-IDF)", LinearSVC(class_weight=PESOS_CLASE, C=1.0))
    ]

    for nombre, modelo in modelos:
        f1 = entrenar_y_evaluar(modelo, X_train_tfidf, y_train, X_test_tfidf, y_test, nombre)
        resultados.append((nombre, f1))

    print("\n=== Comparación de modelos (Macro-F1) ===")
    print(pd.DataFrame(resultados, columns=["Modelo", "Macro-F1"]).sort_values("Macro-F1", ascending=False))
    print("\n✅ Proceso completo finalizado.")

