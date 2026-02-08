# ---------------------------------------------------------------------------
# Custom errors
# ---------------------------------------------------------------------------


class LowConfidenceError(ValueError):
    """Raised when the model's confidence is below the required threshold."""

    def __init__(
        self, label: str, confidence: float, threshold: float
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Model confidence {confidence:.4f} for label '{label}' "
            f"is below the required threshold {threshold:.4f}."
        )