from transformers import pipeline

# Force PyTorch backend explicitly
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)

def analyze_sentiment(text: str):
    if not text or text.strip() == "":
        return None

    result = sentiment_pipeline(text)[0]

    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }
