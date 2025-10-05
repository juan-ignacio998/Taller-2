import README as project


def test_preprocess_text_normalizes_and_filters():
    tokens = project.preprocess_text("Que buen servicio!!! 😃 xq llegó rápido")
    # "que" debe normalizarse y filtrarse como stopword
    assert "que" not in tokens
    assert "buen" in tokens
    assert "servicio" in tokens
    assert "rapido" in tokens or "rápido" in tokens


def test_analyze_sentiment_accepts_custom_analyzer():
    stub_outputs = [
        {"label": "POS", "score": 0.9},
        {"label": "NEG", "score": 0.2},
    ]

    def stub_analyzer(reviews, truncation=True):
        assert truncation is True
        return stub_outputs[: len(reviews)]

    labels, scores = project.analyze_sentiment(
        ["Me encantó", "No me gustó"], analyzer=stub_analyzer
    )

    assert labels == ["pos", "neg"]
    assert scores == [0.9, 0.2]


def test_load_spanish_stopwords_fallback(monkeypatch):
    monkeypatch.setattr(project, "stopwords", None)

    fallback = project.load_spanish_stopwords()

    assert isinstance(fallback, set)
    assert {"de", "la", "que"}.issubset(fallback)
