from transformers import AutoModelForSequenceClassification

def load_kobert_model(num_labels=2, cache_dir="./hf_cache"):
    return AutoModelForSequenceClassification.from_pretrained(
        "monologg/kobert",
        num_labels=num_labels,
        trust_remote_code=True,
        cache_dir=cache_dir
    )