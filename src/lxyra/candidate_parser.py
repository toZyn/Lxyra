"""Opt-in, checksum-verified candidate detector ensemble."""

from pathlib import Path

import onnxruntime as ort
from PIL import Image, ImageOps

from .pipeline import ScreenParser
from .promotion import combine_documents, load_policy
from .reranking import infer_v4


class ValidatedCandidateParser:
    """Keep baseline predictions for every class without validated promotion."""

    def __init__(
        self,
        candidate_model: str | Path,
        promotion_policy: str | Path,
        model_directory=None,
        providers=None,
    ):
        self.baseline = ScreenParser(model_directory=model_directory, providers=providers)
        self.policy = load_policy(promotion_policy, candidate_model)
        self.candidate_session = ort.InferenceSession(
            str(candidate_model),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )

    def predict(self, image: Image.Image, threshold: float = 0.15, tiled: bool = False):
        image = ImageOps.exif_transpose(image).convert("RGB")
        original = self.baseline.predict(image, threshold=threshold, tiled=tiled)
        categories = self.policy["candidate_categories"]
        if not categories:
            return original
        candidate = infer_v4(
            image,
            self.candidate_session,
            self.baseline.text,
            self.baseline.reranking,
            threshold=threshold,
            tiled=tiled,
        )
        return combine_documents(original, candidate, categories)
