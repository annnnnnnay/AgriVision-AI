"""
TensorFlow Model Service — handles model loading, preprocessing, and inference.

This is the heart of AgriVision AI. It loads a fine-tuned MobileNetV2 model
and provides fast, reliable predictions for crop disease detection.
"""

import json
import logging
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
BACKEND_DIR = Path(__file__).resolve().parents[2]


class ModelService:
    """Manages the TensorFlow model lifecycle and inference."""

    def __init__(self):
        self._model = None
        self._class_names: List[str] = []
        self._class_mapping: Dict[int, Dict] = {}
        self._is_loaded = False
        self._input_size = (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        self._model_version = settings.MODEL_VERSION

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        """Load the TensorFlow model and class mapping from disk."""
        if os.getenv("SKIP_TF_LOAD") == "true" and not settings.LOCAL_DEMO_MODE:
            logger.info("SKIP_TF_LOAD is true. Skipping local TensorFlow model loading to save RAM.")
            self._load_demo_mode()
            return

        try:
            import tensorflow as tf

            model_path = settings.MODEL_DIR / settings.MODEL_NAME
            class_mapping_path = BACKEND_DIR / "data" / "class_mapping.json"

            if not model_path.exists():
                if settings.LOCAL_DEMO_MODE:
                    raise FileNotFoundError(f"Trained model not found at {model_path}.")
                logger.warning(
                    f"Model file not found at {model_path}. "
                    "Cloud inference will be used if configured."
                )
                self._load_demo_mode()
                return

            # Load model with optimized settings
            self._model = tf.keras.models.load_model(
                str(model_path),
                compile=False,  # No need to compile for inference
            )

            input_shape = self._model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            height, width = input_shape[1:3]
            if height is not None and width is not None:
                self._input_size = (int(height), int(width))

            # Warm up the model with a dummy prediction
            dummy_input = np.random.rand(1, *self._input_size, 3).astype(np.float32)
            self._model.predict(dummy_input, verbose=0)

            # Load class mapping
            if not class_mapping_path.exists():
                raise FileNotFoundError(
                    f"Model class mapping not found at {class_mapping_path}."
                )
            with open(class_mapping_path, "r") as f:
                raw_mapping = json.load(f)
                self._class_mapping = {
                    int(k): v for k, v in raw_mapping.items()
                }
                output_count = int(self._model.output_shape[-1])
                if len(self._class_mapping) != output_count:
                    raise ValueError(
                        f"Model has {output_count} outputs but class mapping contains "
                        f"{len(self._class_mapping)} classes."
                    )
                self._class_names = [
                    self._class_mapping[i]["name"]
                    for i in sorted(self._class_mapping.keys())
                ]

            self._is_loaded = True
            logger.info(
                f"Model loaded successfully. "
                f"Classes: {len(self._class_names)}, "
                f"Input shape: {self._model.input_shape}"
            )

        except ImportError:
            logger.error("TensorFlow is not installed; local model inference is unavailable.")
            if settings.LOCAL_DEMO_MODE:
                raise RuntimeError("TensorFlow is required when LOCAL_DEMO_MODE is enabled.")
            self._load_demo_mode()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if settings.LOCAL_DEMO_MODE:
                raise RuntimeError("Failed to load the trained local model.") from e
            self._load_demo_mode()

    def _load_demo_mode(self) -> None:
        """Load class labels when inference is delegated to a cloud provider."""
        logger.info("No local model loaded; cloud inference is required.")
        # Load class names from mapping if available
        class_mapping_path = BACKEND_DIR / "data" / "class_mapping.json"
        if class_mapping_path.exists():
            with open(class_mapping_path, "r") as f:
                raw_mapping = json.load(f)
                self._class_mapping = {
                    int(k): v for k, v in raw_mapping.items()
                }
                self._class_names = [
                    self._class_mapping[i]["name"]
                    for i in sorted(self._class_mapping.keys())
                ]
        else:
            self._class_names = []
        self._is_loaded = False

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for model inference.

        Steps:
        1. Resize to the saved model's input size (160x160)
        2. Convert to RGB (handles RGBA, grayscale, etc.)
        3. Leave pixel values in the model-native scale
        4. Add batch dimension
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to the exact trained model input shape.
        image = image.resize(
            (self._input_size[1], self._input_size[0]),
            Image.Resampling.LANCZOS,
        )

        # Keep raw pixel values and rely on the model's Rescaling layer.
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_gemini(
        self,
        image: Image.Image,
        language: str = "en",
        image_type: str = "unknown",
        crop_hint: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Run inference using the Gemini Flash Vision API.
        """
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            logger.warning("GEMINI_API_KEY not configured. Skipping Gemini prediction.")
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            img_copy = self._prepare_cloud_image(image)

            model = genai.GenerativeModel(settings.GEMINI_MODEL)

            prompt = self._build_vision_prompt(language, image_type, crop_hint)

            logger.info("Sending image to Gemini %s (lang: %s)...", settings.GEMINI_MODEL, language)
            start_time = time.perf_counter()
            response = model.generate_content(
                [img_copy, prompt],
                generation_config={"response_mime_type": "application/json"},
                request_options={
                    "timeout": settings.PREDICTION_PROVIDER_TIMEOUT_SECONDS,
                    "retry": None,
                },
            )

            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            data = json.loads(response.text.strip())
            return self._build_vision_result(
                data, processing_time_ms, settings.GEMINI_MODEL, crop_hint
            )

        except Exception as e:
            logger.error(f"Gemini prediction failed, falling back to OpenRouter/Local model: {e}", exc_info=True)
            return None

    def predict_openrouter(
        self,
        image: Image.Image,
        language: str = "en",
        image_type: str = "unknown",
        crop_hint: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Run inference using OpenRouter Vision API with Gemini or Llama vision.
        """
        api_key = settings.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not configured. Skipping OpenRouter prediction.")
            return None

        try:
            import httpx
            import base64
            from io import BytesIO

            img_copy = self._prepare_cloud_image(image)

            # Convert to base64
            buffered = BytesIO()
            img_copy.save(buffered, format="JPEG", quality=85, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            prompt = self._build_vision_prompt(language, image_type, crop_hint)

            logger.info(f"Sending image to OpenRouter Vision API (lang: {language})...")
            start_time = time.perf_counter()

            model_name = f"google/{settings.GEMINI_MODEL}"
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://agrivision.ai",
                "X-Title": "AgriVision AI"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 1000
            }

            timeout = settings.PREDICTION_PROVIDER_TIMEOUT_SECONDS
            fallback_model = "openrouter/free"
            models_to_try = [model_name, fallback_model]
            for attempt, candidate_model in enumerate(models_to_try):
                payload["model"] = candidate_model
                res = httpx.post(url, headers=headers, json=payload, timeout=timeout)
                if res.status_code in {400, 402, 404} and attempt == 0:
                    logger.warning(
                        "OpenRouter model %s returned %d; trying free vision routing.",
                        candidate_model,
                        res.status_code,
                    )
                    continue
                if res.status_code != 200:
                    logger.warning(
                        "OpenRouter request failed with status %d.", res.status_code
                    )
                    return None

                response_data = res.json()
                content_text = response_data["choices"][0]["message"]["content"]
                try:
                    data = json.loads(content_text.strip())
                except json.JSONDecodeError:
                    if attempt == 0:
                        logger.warning(
                            "OpenRouter model %s returned invalid JSON; "
                            "trying free vision routing.",
                            candidate_model,
                        )
                        continue
                    raise
                break
            else:
                return None

            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_vision_result(
                data,
                processing_time_ms,
                f"openrouter-{response_data.get('model', settings.GEMINI_MODEL)}",
                crop_hint,
            )

        except Exception as e:
            logger.error(f"OpenRouter prediction failed, falling back to local model: {e}", exc_info=True)
            return None

    def predict(
        self,
        image: Image.Image,
        language: str = "en",
        image_type: str = "unknown",
        crop_hint: Optional[str] = None,
    ) -> Dict:
        """
        Try cloud vision providers first, then fall back to the local leaf classifier.
        """
        if image_type not in {"leaf", "fruit", "other", "unknown"}:
            raise ValueError("image_type must be leaf, fruit, other, or unknown.")

        canonical_crop_hint = self._canonicalize_crop_hint(crop_hint)
        gemini_result = self.predict_gemini(
            image, language, image_type, canonical_crop_hint
        )
        if gemini_result is not None:
            if self._matches_crop_hint(gemini_result, canonical_crop_hint):
                logger.info("Prediction completed with Gemini (tier 1).")
                return gemini_result
            logger.warning("Gemini result conflicts with the selected crop; trying the next tier.")

        openrouter_result = self.predict_openrouter(
            image, language, image_type, canonical_crop_hint
        )
        if openrouter_result is not None:
            if self._matches_crop_hint(openrouter_result, canonical_crop_hint):
                logger.info("Prediction completed with OpenRouter (tier 2).")
                return openrouter_result
            logger.warning("OpenRouter result conflicts with the selected crop.")

        if image_type != "leaf":
            raise RuntimeError(
                "Cloud vision is unavailable or could not verify this image. "
                "The local model supports leaf photos only; configure a vision provider "
                "for fruit, other, or unspecified images."
            )

        if self._model is None:
            raise RuntimeError(
                "Cloud vision is unavailable, and no local model is loaded."
            )

        logger.info("Cloud providers unavailable; using the local model (tier 3).")
        return self.predict_local(image, crop_hint=canonical_crop_hint)

    @staticmethod
    def _prepare_cloud_image(image: Image.Image) -> Image.Image:
        from io import BytesIO

        prepared = image.convert("RGB")
        prepared.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        prepared.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.copy()

    @staticmethod
    def _build_vision_prompt(
        language: str, image_type: str, crop_hint: Optional[str]
    ) -> str:
        lang_name = {"en": "English", "hi": "Hindi", "mr": "Marathi"}.get(
            language, "English"
        )
        part = {
            "leaf": "The user says the image shows a leaf.",
            "fruit": "The user says the image shows a fruit.",
            "other": "The user says the image shows another plant part.",
            "unknown": "The plant part is not specified; inspect the visible subject.",
        }[image_type]
        hint = (
            f"The user identifies the crop as {crop_hint}. Verify this visually; "
            "if the image does not support that crop, return Unknown rather than "
            "substituting another crop."
            if crop_hint
            else "Identify the crop from visible evidence; do not guess from color alone."
        )
        return f"""
You are an agricultural image triage assistant. Inspect the visible plant part before
identifying a crop or disease. {part} {hint}
For fruit, use multiple visible features rather than color alone to distinguish crops.
Do not label an apple as tomato or strawberry without clear visual evidence; if unsure,
return Unknown.
When the image shows fruit, do not diagnose a leaf-only disease from the fruit.
If the crop or disease is not visually identifiable, use "Unknown" and low confidence;
do not invent symptoms, disease, or treatment. Do not provide treatment advice for an
uncertain diagnosis.

Return one valid JSON object with this schema:
{{
  "crop": "English crop name, or Unknown",
  "disease": "English disease name, Healthy, or Unknown",
  "disease_id": "English snake_case identifier, or unknown",
  "confidence": 0.0,
  "is_healthy": false,
  "severity": null,
  "description": "Localized in {lang_name}; empty when uncertain",
  "symptoms": [],
  "causes": [],
  "treatment": {{"organic": [], "chemical": []}},
  "prevention": [],
  "severity_info": null
}}
Keep crop, disease, and disease_id in English. Localize only descriptive fields
and advice in {lang_name}. Confidence must be a number from 0 to 1 reflecting visual
evidence, not certainty in general knowledge.
"""

    def _build_vision_result(
        self,
        data: Dict,
        processing_time_ms: float,
        model_version: str,
        crop_hint: Optional[str],
    ) -> Optional[Dict]:
        if not isinstance(data, dict):
            return None

        crop = str(data.get("crop", "Unknown")).strip() or "Unknown"
        disease = str(data.get("disease", "Unknown")).strip() or "Unknown"
        if crop_hint and not self._crop_names_match(crop, crop_hint):
            logger.warning(
                "Vision provider returned crop %r despite selected crop %r.",
                crop,
                crop_hint,
            )
            return None
        if crop_hint:
            crop = crop_hint

        try:
            confidence = float(data["confidence"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Vision provider returned no valid confidence score.")
            return None
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            logger.warning("Vision provider returned an out-of-range confidence score.")
            return None

        if crop.casefold() == "unknown" or disease.casefold() == "unknown":
            logger.info("Vision provider abstained because crop or disease was unclear.")
            return None

        is_healthy = data.get("is_healthy") is True
        predicted_class = f"{crop}___{'healthy' if is_healthy else self._slug(disease)}"
        severity = data.get("severity") if not is_healthy else None
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "crop": crop,
            "is_healthy": is_healthy,
            "severity": severity,
            "top_predictions": [{"class": predicted_class, "confidence": confidence}],
            "processing_time_ms": processing_time_ms,
            "model_version": model_version,
            "gemini_rich_info": {
                "name": disease,
                "crop": crop,
                "description": data.get("description", ""),
                "symptoms": data.get("symptoms", []),
                "causes": data.get("causes", []),
                "treatment": data.get("treatment", {}),
                "prevention": data.get("prevention", []),
                "severity_info": data.get("severity_info"),
            },
        }

    def _canonicalize_crop_hint(self, crop_hint: Optional[str]) -> Optional[str]:
        if crop_hint is None or not crop_hint.strip():
            return None

        normalized_hint = self._normalize_crop_name(crop_hint)
        if len(normalized_hint) < 3:
            raise ValueError("Crop hint must contain at least three letters.")

        supported_crops = {
            str(info.get("crop") or self._extract_crop_name(info.get("name", "")))
            for info in self._class_mapping.values()
        }
        matches = [
            crop
            for crop in supported_crops
            if self._normalize_crop_name(crop).startswith(normalized_hint)
        ]
        if len(matches) != 1:
            choices = ", ".join(sorted(supported_crops))
            raise ValueError(f"Crop hint must match one supported crop: {choices}.")
        return matches[0]

    @staticmethod
    def _normalize_crop_name(crop: str) -> str:
        return re.sub(r"[^a-z0-9]", "", crop.casefold())

    @classmethod
    def _crop_names_match(cls, left: str, right: str) -> bool:
        normalized_left = cls._normalize_crop_name(left)
        normalized_right = cls._normalize_crop_name(right)
        return (
            normalized_left == normalized_right
            or normalized_left.startswith(normalized_right)
            or normalized_right.startswith(normalized_left)
        )

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")

    @classmethod
    def _matches_crop_hint(
        cls, prediction: Dict, crop_hint: Optional[str]
    ) -> bool:
        crop = str(prediction.get("crop", "Unknown"))
        return (
            not crop_hint
            or crop.casefold() == "unknown"
            or cls._crop_names_match(crop, crop_hint)
        )

    def predict_local(
        self, image: Image.Image, crop_hint: Optional[str] = None
    ) -> Dict:
        """
        Run inference on an image and return predictions.

        Returns:
            {
                "predicted_class": str,
                "confidence": float,
                "top_predictions": [{"class": str, "confidence": float}, ...],
                "processing_time_ms": float,
            }
        """
        start_time = time.perf_counter()

        # Preprocess
        processed = self.preprocess_image(image)

        if self._model is None:
            raise RuntimeError("No trained local model is loaded.")

        predictions = self._model.predict(processed, verbose=0)[0]

        if crop_hint:
            crop_indices = [
                index
                for index, info in self._class_mapping.items()
                if index < len(predictions)
                and self._crop_names_match(
                    str(info.get("crop") or self._extract_crop_name(info.get("name", ""))),
                    crop_hint,
                )
            ]
            if not crop_indices:
                raise RuntimeError(
                    f"The local model has no classes for the selected crop {crop_hint}."
                )
            crop_indices.sort(key=lambda index: predictions[index], reverse=True)
        else:
            # Select the crop from its combined class probability, then rank diseases within it.
            _, crop_indices = self._select_crop_predictions(
                predictions, self._class_mapping, image
            )
        top_indices = crop_indices[:3]
        top_predictions = []
        for idx in top_indices:
            class_name = (
                self._class_mapping.get(idx, {}).get("name", f"class_{idx}")
                if self._class_mapping
                else self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
            )
            top_predictions.append({
                "class": class_name,
                "confidence": round(float(predictions[idx]), 4),
            })

        # Best prediction
        best_class = top_predictions[0]["class"]
        best_confidence = top_predictions[0]["confidence"]
        if best_confidence < settings.CONFIDENCE_THRESHOLD:
            raise RuntimeError(
                "The local model could not make a confident leaf diagnosis. "
                "Use a clearer leaf photo or configure a vision provider."
            )

        # Extract crop name from the selected crop's own disease class.
        crop_name = self._extract_crop_name(best_class)

        # Determine if healthy
        is_healthy = "healthy" in best_class.lower()

        # Determine severity (only for diseased)
        severity = self._assess_severity(best_confidence, is_healthy)

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "predicted_class": best_class,
            "confidence": best_confidence,
            "crop": crop_name,
            "is_healthy": is_healthy,
            "severity": severity,
            "top_predictions": top_predictions,
            "processing_time_ms": processing_time_ms,
            "model_version": self._model_version,
        }

    @staticmethod
    def _select_crop_predictions(
        predictions: np.ndarray,
        class_mapping: Dict[int, Dict],
        image: Optional[Image.Image] = None,
    ) -> Tuple[int, List[int]]:
        """Choose a crop by total score, correcting obvious shape mismatches."""
        if not class_mapping:
            ranked_indices = np.argsort(predictions)[::-1].tolist()
            return ranked_indices[0], ranked_indices

        crop_indices: Dict[str, List[int]] = {}
        for index, class_info in class_mapping.items():
            if index >= len(predictions):
                continue
            crop = class_info.get("crop")
            if not crop:
                class_name = class_info.get("name", "")
                crop = ModelService._extract_crop_name(class_name)
            if crop:
                crop_indices.setdefault(crop, []).append(index)

        if len(crop_indices) < 2:
            ranked_indices = np.argsort(predictions)[::-1].tolist()
            return ranked_indices[0], ranked_indices

        crop_scores = {
            crop: float(np.sum(predictions[indices]))
            for crop, indices in crop_indices.items()
        }
        selected_crop = max(crop_scores, key=crop_scores.get)
        ranked_indices = sorted(
            crop_indices[selected_crop],
            key=lambda index: predictions[index],
            reverse=True,
        )

        if image is not None:
            class_name = class_mapping[ranked_indices[0]].get("name", "")
            inferred_crop = ModelService._infer_crop_from_leaf_shape(
                image, selected_crop, class_name
            )
            matching_crop = next(
                (
                    crop
                    for crop in crop_indices
                    if crop.casefold() == inferred_crop.casefold()
                ),
                None,
            )
            if matching_crop is not None and matching_crop != selected_crop:
                selected_crop = matching_crop
                ranked_indices = sorted(
                    crop_indices[selected_crop],
                    key=lambda index: predictions[index],
                    reverse=True,
                )

        return ranked_indices[0], ranked_indices

    @staticmethod
    def _assess_severity(confidence: float, is_healthy: bool) -> Optional[str]:
        """Assess disease severity based on confidence score."""
        if is_healthy:
            return None
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.60:
            return "moderate"
        else:
            return "low"

    @staticmethod
    def _extract_crop_name(class_name: str) -> str:
        """Extract crop name from class string like 'Tomato___Early_blight'."""
        if "___" in class_name:
            return class_name.split("___")[0]
        return class_name.split("_")[0] if "_" in class_name else class_name

    @staticmethod
    def _infer_crop_from_leaf_shape(image: Image.Image, fallback_crop: str, class_name: str) -> str:
        """Correct shared-disease class confusion (e.g. tomato vs potato) using leaf shape complexity."""
        try:
            crop = (fallback_crop or "").strip()
            crop_lower = crop.lower()
            if crop_lower not in {"tomato", "potato", "strawberry", "corn_(maize)", "corn", "maize", "mango"}:
                return crop or "Unknown"

            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            green_mask = (
                (rgb[:, :, 1] > rgb[:, :, 0])
                & (rgb[:, :, 1] > rgb[:, :, 2])
                & (rgb[:, :, 1] > 60)
            )
            if green_mask.mean() < 0.05:
                return crop

            ys, xs = np.where(green_mask)
            if xs.size == 0:
                return crop

            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            leaf = green_mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
            if leaf.size == 0:
                return crop

            edge = (
                (leaf[1:, 1:] != leaf[:-1, :-1])
                | (leaf[:-1, 1:] != leaf[1:, :-1])
                | (leaf[1:, :-1] != leaf[:-1, 1:])
                | (leaf[:-1, :-1] != leaf[1:, 1:])
            )
            boundary_ratio = edge.sum() / max(leaf.sum(), 1)
            aspect_ratio = (x1 - x0 + 1) / max(y1 - y0 + 1, 1)

            if crop_lower in {"corn_(maize)", "corn", "maize"}:
                coordinates = np.column_stack((xs, ys))
                eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
                elongation = np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-6))
                if (
                    0.85 <= aspect_ratio <= 1.8
                    and leaf.mean() >= 0.35
                    and elongation <= 2.2
                ):
                    return "Mango"
                return crop

            if crop_lower == "mango":
                return crop

            # Use a generalized leaf-shape profile so that a potato leaf is not mistaken for
            # strawberry or tomato when the model predicts a wrong crop/disease pairing.
            # Real leaf silhouettes are noisy, so the width/height ratio is the more stable cue.
            if aspect_ratio < 0.82:
                shape_group = "potato_like"
            elif aspect_ratio > 0.85:
                shape_group = "tomato_like"
            else:
                shape_group = "strawberry_like" if boundary_ratio <= 0.25 else "tomato_like"

            disease_label = (class_name or "").lower()
            is_shared_disease = any(token in disease_label for token in ["early_blight", "late_blight", "leaf_spot", "leaf_scorch", "blight"])

            if crop_lower == "strawberry" and shape_group == "potato_like":
                return "Potato"
            if crop_lower == "potato" and shape_group == "tomato_like":
                return "Tomato"
            if crop_lower == "tomato" and shape_group == "potato_like":
                return "Potato"
            if crop_lower == "tomato" and shape_group == "strawberry_like":
                return "Strawberry"
            if crop_lower == "strawberry" and shape_group == "tomato_like":
                return "Tomato"

            if crop_lower == "potato" and is_shared_disease and shape_group == "tomato_like":
                return "Tomato"
            if crop_lower == "tomato" and is_shared_disease and shape_group == "potato_like":
                return "Potato"

            return crop
        except Exception:
            return fallback_crop
