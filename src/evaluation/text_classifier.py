"""Off-the-shelf classifiers for evaluating generated text."""

from transformers import pipeline


CLASSIFIER_MAP: dict[str, str] = {
    "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "toxicity": "unitary/toxic-bert",
    "civil_comments": "unitary/toxic-bert",
    "truthfulqa": "distilbert-base-uncased-finetuned-sst-2-english",  # fallback
    "truthfulness": "distilbert-base-uncased-finetuned-sst-2-english",
}


class TextClassifier:
    """Wrapper around a HuggingFace text-classification pipeline."""

    def __init__(self, classifier_name: str, device: str = "cpu") -> None:
        """Initialize the text classifier.

        Args:
            classifier_name: HuggingFace model identifier for the classifier.
            device: Device string, either "cpu" or "cuda".
        """
        device_int = 0 if device == "cuda" else -1
        self.pipe = pipeline(
            "text-classification",
            model=classifier_name,
            device=device_int,
        )
        self.classifier_name = classifier_name

    def classify(self, texts: list[str]) -> list[dict]:
        """Classify a list of texts.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of dicts, each with 'label' and 'score' keys.
        """
        results = self.pipe(texts, truncation=True)
        return [{"label": r["label"], "score": r["score"]} for r in results]

    def classify_positive_rate(
        self, texts: list[str], positive_label: str = "POSITIVE"
    ) -> float:
        """Compute the fraction of texts classified as the positive label.

        Args:
            texts: List of text strings to classify.
            positive_label: Label string considered as positive.

        Returns:
            Fraction of texts classified as positive_label.
        """
        if len(texts) == 0:
            return 0.0
        results = self.classify(texts)
        n_positive = sum(1 for r in results if r["label"] == positive_label)
        return n_positive / len(texts)


def get_classifier(dataset_name: str, device: str = "cpu") -> TextClassifier:
    """Get a TextClassifier for the given dataset.

    Args:
        dataset_name: Name of the dataset (key in CLASSIFIER_MAP).
        device: Device string, either "cpu" or "cuda".

    Returns:
        A TextClassifier instance.

    Raises:
        KeyError: If dataset_name is not found in CLASSIFIER_MAP.
    """
    if dataset_name not in CLASSIFIER_MAP:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(CLASSIFIER_MAP.keys())}"
        )
    return TextClassifier(CLASSIFIER_MAP[dataset_name], device=device)
