# ---------------------------------------------------------------------------
# Custom errors
# ---------------------------------------------------------------------------


class UnknownLabelError(ValueError):
    """Raised when the model returns a label not in the expected mapping."""

    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence
        super().__init__(
            f"Model returned unknown sentiment label '{label}' "
            f"(confidence={confidence:.4f}).  Expected one of: "
            f"positive, negative, neutral."
        )