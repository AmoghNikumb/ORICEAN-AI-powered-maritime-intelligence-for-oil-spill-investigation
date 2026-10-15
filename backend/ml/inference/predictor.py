"""
predictor.py — OilSpillPredictor and TilePrediction for ORICEAN.

Loads the ``MeghanaK25/sar-oil-slick-detection`` model (a torchvision
ResNet-50 binary classifier fine-tuned on Sentinel-1 SAR tiles) and exposes
``predict()`` / ``predict_batch()`` for SAR tile inference.

Model contract (verified against the published weights, not just the card):
- ``model.pth`` is a **state_dict** (OrderedDict), keys prefixed ``resnet.`` —
  i.e. the training code wrapped ``torchvision.models.resnet50`` as
  ``self.resnet``. The prefix is stripped and loaded into a plain resnet50.
- ``resnet.conv1.weight`` is ``(64, 3, 7, 7)`` → the network takes a
  **3-channel** 224×224 input (the card's ``num_channels: 1`` is inaccurate;
  the weights are authoritative).
- ``resnet.fc`` is ``Linear(2048, 2)`` → 2-class logits, trained with
  CrossEntropyLoss. Oil-slick probability = ``softmax(logits, dim=1)[:, 1]``.
- Normalization: standard 3-channel ImageNet mean/std. This was picked
  empirically — it is the only variant that produces a discriminative
  response (dark SAR regions → high p_oil), and it matches the preprocessing
  the ORICEAN pipeline already applied to the previous MobileNetV3 model.
- Recommended decision threshold: 0.5 (``config.json`` ``recommended_threshold``).

Usage (called once at FastAPI startup via lifespan):

    predictor = OilSpillPredictor(
        model_path=settings.MODEL_PATH,
        device=settings.INFERENCE_DEVICE or None,
        config_path=settings.MODEL_CONFIG_PATH or None,
    )
    predictor.load()

``config.py`` is intentionally NOT imported here; all configuration is
injected through ``__init__`` parameters so this module stays testable
in isolation.
"""

import io
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TilePrediction:
    """Structured result returned by a single-tile inference call.

    Attributes:
        label:       Human-readable class name — ``"Oil Spill"`` or ``"No Oil"``.
        confidence:  Probability of the positive (oil-slick) class in [0, 1],
                     i.e. ``softmax(logits)[1]`` straight from the model.
        is_positive: ``True`` when ``confidence >= predictor.confidence_threshold``.
        logits:      ``[p_no_oil, p_oil]`` softmax probabilities (length 2).
        mock:        ``True`` when the prediction was synthesised (mock mode).
    """

    label: str
    confidence: float
    is_positive: bool
    logits: list[float]
    mock: bool = False


# ---------------------------------------------------------------------------
# Preprocessing constants — 3-channel ImageNet normalization
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class OilSpillPredictor:
    """Loads the SAR oil-slick ResNet-50 classifier and runs SAR-tile inference.

    The predictor is designed to be initialised **once** at application
    startup and kept alive for the lifetime of the process.  Call
    ``load()`` after construction to place the model on the target device.

    Args:
        model_path:           Path to the ``.pth`` state_dict on disk.
        model_name:           torchvision architecture name
                              (default: ``"resnet50"`` — matches the weights).
        num_classes:          Number of output classes (default: ``2``).
        device:               PyTorch device string.  Pass ``None`` to
                              auto-detect: CUDA if available, else CPU.
        confidence_threshold: Minimum oil-slick probability to treat a
                              tile as a positive detection (default: ``0.5``,
                              the model's recommended threshold).
        config_path:          Optional path to the model's ``config.json``.
                              Only its ``recommended_threshold`` is honoured;
                              ``num_channels`` / single-value normalization in
                              that file do not match the published weights and
                              are ignored.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "resnet50",
        num_classes: int = 2,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        config_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.config_path = config_path

        # Auto-detect device when caller passes None
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model: Optional[torch.nn.Module] = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the state_dict from disk and prepare the model for inference.

        Must be called once before any ``predict()`` / ``predict_batch()``
        calls.  Safe to call multiple times — subsequent calls reload the
        weights from disk.

        Raises:
            FileNotFoundError: When ``self.model_path`` does not exist.

        On ``RuntimeError`` (e.g. architecture/state-dict mismatch) the
        error is logged at ERROR level and ``is_loaded`` is set to
        ``False``.  The application is **not** crashed — a degraded-mode
        response (``HTTP 503``) is expected from the route layer.
        """
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                "Place the .pth file in backend/ml/models/ and update MODEL_PATH in .env."
            )

        # Honour the recommended decision threshold from config.json when present.
        if self.config_path:
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                rec = cfg.get("deployment", {}).get("recommended_threshold")
                if isinstance(rec, (int, float)):
                    self.confidence_threshold = float(rec)
            except Exception as exc:  # pragma: no cover - config is best-effort
                logger.warning("Could not read model config '%s': %s", self.config_path, exc)

        logger.info(
            "Loading model '%s' from '%s' onto device '%s'.",
            self.model_name,
            self.model_path,
            self.device,
        )

        try:
            model = self._build_architecture()

            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Support raw state dicts and wrapped checkpoint dicts
            # e.g. {"state_dict": OrderedDict(...)} or {"model": OrderedDict(...)}.
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(
                checkpoint["model"], dict
            ):
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, torch.nn.Module):
                # Some checkpoints pickle the whole module.
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint

            state_dict = self._normalize_state_dict_keys(state_dict)

            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as strict_exc:
                logger.warning(
                    "Strict state_dict load failed (%s); retrying with strict=False.",
                    strict_exc,
                )
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(
                        "Non-strict load — missing=%s unexpected=%s", missing, unexpected
                    )

            model.eval()
            model.to(self.device)

            self._model = model
            self._is_loaded = True
            logger.info(
                "Model loaded successfully. arch=%s device=%s is_loaded=%s threshold=%.2f",
                self.model_name,
                self.device,
                self._is_loaded,
                self.confidence_threshold,
            )

        except RuntimeError as exc:
            logger.error(
                "Failed to load model weights from '%s': %s. "
                "The application will continue without ML inference.",
                self.model_path,
                exc,
            )
            self._is_loaded = False
            # Deliberately not re-raised — app must not crash at startup.

    # ------------------------------------------------------------------
    # Architecture / state_dict helpers
    # ------------------------------------------------------------------

    def _build_architecture(self) -> torch.nn.Module:
        """Instantiate an untrained torchvision backbone with a 2-class head."""
        factory = getattr(torchvision.models, self.model_name, None)
        if factory is None:
            raise RuntimeError(
                f"Unknown torchvision architecture '{self.model_name}'."
            )
        model = factory(weights=None)
        # torchvision ResNet variants expose `.fc`; swap in a fresh classifier.
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
            model.classifier = torch.nn.Linear(
                model.classifier.in_features, self.num_classes
            )
        return model

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict) -> dict:
        """Strip common wrapper prefixes so keys line up with a bare resnet50.

        The published weights prefix every key with ``resnet.`` (the training
        code held the backbone as ``self.resnet``). ``module.`` (DataParallel)
        and ``model.`` are also handled defensively.
        """
        prefixes = ("resnet.", "module.", "model.", "backbone.")
        cleaned: dict = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            cleaned[new_key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` after a successful ``load()`` call."""
        return self._is_loaded

    @property
    def model(self) -> Optional[torch.nn.Module]:
        """The loaded ``torch.nn.Module``, or ``None`` if not yet loaded.

        Used by external callers (e.g. ``generate_gradcam``) that need direct
        access to the model for activation-based visualisation.
        """
        return self._model

    @property
    def model_version(self) -> str:
        """Human-readable identifier for the loaded weights.

        Returns ``"unknown"`` when the predictor has not been loaded yet.
        Uses the parent directory name when the file is generically named
        (e.g. ``model.pth``), otherwise the filename stem.
        """
        if not self._is_loaded:
            return "unknown"
        p = Path(self.model_path)
        stem = p.stem
        if stem.lower() in ("model", "weights", "best", "checkpoint", "pytorch_model"):
            return f"{p.parent.name}_{self.model_name}"
        return stem

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> TilePrediction:
        """Run inference on a single SAR tile.

        Convenience wrapper around ``predict_batch()`` for single-tile use.

        Args:
            image_bytes: Raw bytes of a PNG/JPEG/TIFF image tile.

        Returns:
            A :class:`TilePrediction` with label, confidence, and probabilities.
        """
        return self.predict_batch([image_bytes])[0]

    def predict_batch(self, images: list[bytes]) -> list[TilePrediction]:
        """Run inference on a batch of SAR tile image bytes.

        Supports two execution paths controlled by the ``ML_MOCK_MODE``
        environment variable:

        * **Mock mode** (``ML_MOCK_MODE=true``): returns synthetic predictions
          without loading or running the real model.  Useful for development
          and integration testing.
        * **Real inference**: decodes each image, applies the model's
          preprocessing (grayscale → 3-channel → Resize 224×224 →
          scale to [0, 1] → ImageNet Normalize), stacks into a batch, and runs
          a forward pass through the ResNet-50 classifier. The oil-slick
          confidence is ``softmax(logits, dim=1)[:, 1]`` — nothing is added,
          boosted, or synthesised.

        Invalid/unreadable image bytes are silently skipped with a WARNING
        log; the corresponding entry is omitted from the returned list.

        Args:
            images: List of raw image bytes (PNG, JPEG, TIFF, etc.).

        Returns:
            List of :class:`TilePrediction` objects, one per *valid* input
            image.  The list length may be shorter than ``len(images)`` if
            any tiles failed to decode.
        """
        mock_mode = os.environ.get("ML_MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            return self._predict_batch_mock(images)
        return self._predict_batch_real(images)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_batch_mock(self, images: list[bytes]) -> list[TilePrediction]:
        """Return synthetic predictions (mock mode)."""
        results: list[TilePrediction] = []
        for image_bytes in images:
            # Still attempt to decode so obviously corrupt bytes are caught.
            try:
                Image.open(io.BytesIO(image_bytes))
            except UnidentifiedImageError:
                logger.warning(
                    "Mock mode: skipping unidentifiable image (%d bytes).",
                    len(image_bytes),
                )
                continue

            confidence = random.uniform(0, 1)
            is_positive = confidence >= self.confidence_threshold
            label = "Oil Spill" if is_positive else "No Oil"
            results.append(
                TilePrediction(
                    label=label,
                    confidence=confidence,
                    is_positive=is_positive,
                    logits=[1.0 - confidence, confidence],
                    mock=True,
                )
            )
        return results

    def _predict_batch_real(self, images: list[bytes]) -> list[TilePrediction]:
        """Run ResNet-50 inference and return the raw model probabilities.

        Preprocessing matches what the model was trained on:
          1. decode → grayscale (SAR is single-band) → replicate to 3 channels
          2. resize to 224×224 (bilinear)
          3. scale to [0, 1], then normalize with 3-channel ImageNet mean/std

        The oil-slick confidence is ``softmax(logits)[1]`` directly. No
        physics heuristics, darkness boosts, or threshold nudging are applied —
        the pipeline's ``confidence_threshold`` alone decides ``is_positive``.
        """
        if self._model is None:
            logger.error("predict_batch called before load(); returning no predictions.")
            return []

        # Manual PIL → numpy → torch conversion. torchvision's ToTensor() uses
        # the numpy C array API which can break inside asyncio worker threads on
        # some numpy/torch builds, so we avoid it here.
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

        tensors: list[torch.Tensor] = []
        for image_bytes in images:
            try:
                img = Image.open(io.BytesIO(image_bytes))
            except UnidentifiedImageError:
                logger.warning("Skipping unidentifiable image (%d bytes).", len(image_bytes))
                continue

            # SAR tiles are single-band; go through L so multi-band / palette
            # inputs collapse consistently, then fan out to 3 identical channels.
            img_rgb = img.convert("L").resize((224, 224), Image.BILINEAR).convert("RGB")

            arr = np.array(img_rgb, dtype=np.float32) / 255.0   # [H, W, 3] in [0, 1]
            tensor = torch.from_numpy(arr).permute(2, 0, 1)      # [3, H, W]
            tensor = (tensor - mean) / std
            tensors.append(tensor)

        if not tensors:
            return []

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self._model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.nn.functional.softmax(logits, dim=1)

        results: list[TilePrediction] = []
        for i in range(probs.shape[0]):
            p_oil = float(probs[i, 1].item())
            p_no_oil = float(probs[i, 0].item())
            is_positive = p_oil >= self.confidence_threshold
            results.append(
                TilePrediction(
                    label="Oil Spill" if is_positive else "No Oil",
                    confidence=p_oil,
                    is_positive=is_positive,
                    logits=[p_no_oil, p_oil],
                    mock=False,
                )
            )
        return results
