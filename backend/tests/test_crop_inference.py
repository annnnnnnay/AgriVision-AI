import builtins
import json
import sys
import types

from PIL import Image
import numpy as np
import pytest

from app.services import model_service
from app.services.model_service import ModelService


def test_skip_tensorflow_load_works_without_importing_tensorflow(monkeypatch):
    imported_modules = []
    original_import = builtins.__import__

    def track_tensorflow_import(name, *args, **kwargs):
        if name == "tensorflow":
            imported_modules.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("SKIP_TF_LOAD", "true")
    monkeypatch.setattr(model_service.settings, "LOCAL_DEMO_MODE", False)
    monkeypatch.setattr(builtins, "__import__", track_tensorflow_import)

    service = ModelService()
    service.load_model()

    assert not service.is_loaded
    assert imported_modules == []


def test_tomato_leaf_shape_prefers_tomato_for_early_blight():
    service = ModelService()

    # Simulate a lobed tomato-like leaf shape with green tissue and yellow disease spots.
    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    # A tomato-like lobed silhouette.
    mask = ((xx - 80) ** 2 / (52 ** 2) + (yy - 80) ** 2 / (58 ** 2) <= 1)
    # Create rough lobes by subtracting two side indentations.
    mask &= ~(((xx - 55) ** 2 / (18 ** 2) + (yy - 80) ** 2 / (20 ** 2) <= 1) & (xx < 80))
    mask &= ~(((xx - 105) ** 2 / (18 ** 2) + (yy - 80) ** 2 / (20 ** 2) <= 1) & (xx > 80))
    canvas[mask] = [20, 120, 25]
    canvas[::4, ::4] = [220, 220, 80]

    img = Image.fromarray(canvas, 'RGB')
    crop = service._infer_crop_from_leaf_shape(img, 'Potato', 'Potato___Early_blight')

    assert crop == 'Tomato', f'Expected Tomato for a lobed tomato-like leaf, got {crop}'


def test_potato_leaf_shape_avoids_strawberry_confusion():
    service = ModelService()

    # Simulate a broad potato-like leaf silhouette instead of a strawberry-like heart leaf.
    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    mask = ((xx - 80) ** 2 / (46 ** 2) + (yy - 80) ** 2 / (62 ** 2) <= 1)
    canvas[mask] = [20, 120, 25]
    canvas[::4, ::4] = [220, 220, 80]

    img = Image.fromarray(canvas, 'RGB')
    crop = service._infer_crop_from_leaf_shape(img, 'Strawberry', 'Strawberry___Leaf_scorch')

    assert crop == 'Potato', f'Expected Potato for a broad potato-like silhouette, got {crop}'


def test_mango_crop_wins_when_its_combined_score_exceeds_strawberry():
    class_mapping = {
        0: {'name': 'Strawberry___Leaf_scorch', 'crop': 'Strawberry'},
        1: {'name': 'Mango___Anthracnose', 'crop': 'Mango'},
        2: {'name': 'Mango___healthy', 'crop': 'Mango'},
    }
    predictions = np.array([0.40, 0.32, 0.28], dtype=np.float32)

    best_index, ranked_indices = ModelService._select_crop_predictions(
        predictions, class_mapping
    )

    assert class_mapping[best_index]['crop'] == 'Mango'
    assert ranked_indices == [1, 2]


def test_broad_mango_canopy_is_not_reported_as_corn():
    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:160, :160]
    dx = xx - 80
    dy = yy - 80
    mask = np.zeros((160, 160), dtype=bool)
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        along = dx * np.cos(angle) + dy * np.sin(angle)
        across = -dx * np.sin(angle) + dy * np.cos(angle)
        mask |= (along / 72) ** 2 + (across / 24) ** 2 <= 1
    canvas[mask] = [30, 125, 35]
    img = Image.fromarray(canvas, 'RGB')

    crop = ModelService._infer_crop_from_leaf_shape(
        img, 'Corn_(maize)', 'Corn_(maize)___healthy'
    )

    assert crop == 'Mango'

    class_mapping = {
        0: {'name': 'Corn_(maize)___healthy', 'crop': 'Corn_(maize)'},
        1: {'name': 'Mango___Anthracnose', 'crop': 'Mango'},
        2: {'name': 'Mango___healthy', 'crop': 'Mango'},
    }
    predictions = np.array([0.60, 0.15, 0.25], dtype=np.float32)
    best_index, ranked_indices = ModelService._select_crop_predictions(
        predictions, class_mapping, img
    )

    assert best_index == 2
    assert ranked_indices == [2, 1]


def test_narrow_corn_blade_is_not_changed_to_mango():
    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:160, :160]
    mask = ((xx - 80) ** 2 / (18 ** 2) + (yy - 80) ** 2 / (72 ** 2) <= 1)
    canvas[mask] = [30, 125, 35]
    img = Image.fromarray(canvas, 'RGB')

    crop = ModelService._infer_crop_from_leaf_shape(
        img, 'Corn_(maize)', 'Corn_(maize)___healthy'
    )

    assert crop == 'Corn_(maize)'


def test_gemini_is_primary_even_in_local_demo_mode(monkeypatch):
    service = ModelService()
    service._model = object()
    result = {"predicted_class": "apple_scab"}
    monkeypatch.setattr("app.services.model_service.settings.LOCAL_DEMO_MODE", True)
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: result,
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: pytest.fail(
            "OpenRouter should not run after Gemini succeeds"
        ),
    )
    monkeypatch.setattr(
        service,
        "predict_local",
        lambda image, crop_hint=None: pytest.fail(
            "Local inference should not run after Gemini succeeds"
        ),
    )

    prediction = service.predict(Image.new("RGB", (16, 16)), image_type="leaf")

    assert prediction is result


def test_openrouter_is_used_when_gemini_fails(monkeypatch):
    service = ModelService()
    service._model = object()
    monkeypatch.setattr("app.services.model_service.settings.LOCAL_DEMO_MODE", False)
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: None,
    )
    result = {"predicted_class": "apple_scab"}
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: result,
    )
    monkeypatch.setattr(
        service,
        "predict_local",
        lambda image, crop_hint=None: pytest.fail(
            "Local inference should not run after OpenRouter succeeds"
        ),
    )

    assert service.predict(Image.new("RGB", (16, 16)), image_type="leaf") is result


def test_local_model_is_last_fallback(monkeypatch):
    service = ModelService()
    service._model = object()
    monkeypatch.setattr("app.services.model_service.settings.LOCAL_DEMO_MODE", False)
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: None,
    )
    result = {"predicted_class": "Tomato___healthy"}
    monkeypatch.setattr(
        service, "predict_local", lambda image, crop_hint=None: result
    )

    assert service.predict(Image.new("RGB", (16, 16)), image_type="leaf") is result


def test_prediction_fails_instead_of_fabricating_results_without_any_tier(monkeypatch):
    service = ModelService()
    monkeypatch.setattr("app.services.model_service.settings.LOCAL_DEMO_MODE", False)
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: None,
    )

    with pytest.raises(RuntimeError, match="no local model is loaded"):
        service.predict(Image.new("RGB", (16, 16)), image_type="leaf")


def test_local_inference_never_generates_demo_predictions():
    service = ModelService()

    with pytest.raises(RuntimeError, match="No trained local model is loaded"):
        service.predict_local(Image.new("RGB", (16, 16)))


@pytest.mark.parametrize("image_type", ["fruit", "other", "unknown"])
def test_local_leaf_model_is_not_used_for_non_leaf_or_unspecified_images(
    monkeypatch, image_type
):
    service = ModelService()
    service._model = object()
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_local",
        lambda image, crop_hint=None: pytest.fail(
            "The local leaf model must not classify this image."
        ),
    )

    with pytest.raises(RuntimeError, match="supports leaf photos only"):
        service.predict(Image.new("RGB", (16, 16)), image_type=image_type)


def test_crop_hint_rejects_mismatched_cloud_crop_and_blocks_fruit_fallback(monkeypatch):
    service = ModelService()
    service._model = object()
    service._class_mapping = {
        0: {"name": "Apple___Apple_scab", "crop": "Apple"},
        1: {"name": "Strawberry___Leaf_scorch", "crop": "Strawberry"},
    }
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: {
            "crop": "Strawberry",
            "predicted_class": "Strawberry___Leaf_scorch",
        },
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_local",
        lambda image, crop_hint=None: pytest.fail(
            "Fruit images must never fall back to the local leaf model."
        ),
    )

    with pytest.raises(RuntimeError, match="supports leaf photos only"):
        service.predict(
            Image.new("RGB", (16, 16)),
            image_type="fruit",
            crop_hint="Apple",
        )


def test_local_crop_hint_restricts_results_to_the_requested_crop(monkeypatch):
    class FixedModel:
        def predict(self, images, verbose=0):
            return np.array([[0.55, 0.45]], dtype=np.float32)

    service = ModelService()
    service._model = FixedModel()
    service._class_mapping = {
        0: {"name": "Apple___Apple_scab", "crop": "Apple"},
        1: {"name": "Strawberry___Leaf_scorch", "crop": "Strawberry"},
    }
    service._class_names = [
        "Apple___Apple_scab",
        "Strawberry___Leaf_scorch",
    ]
    monkeypatch.setattr(
        service,
        "predict_gemini",
        lambda image, language, image_type, crop_hint: None,
    )
    monkeypatch.setattr(
        service,
        "predict_openrouter",
        lambda image, language, image_type, crop_hint: None,
    )

    result = service.predict(
        Image.new("RGB", (16, 16)),
        image_type="leaf",
        crop_hint="Apple",
    )

    assert result["crop"] == "Apple"
    assert result["predicted_class"] == "Apple___Apple_scab"
    assert all("Strawberry" not in item["class"] for item in result["top_predictions"])


def test_gemini_request_uses_configured_timeout_without_sdk_retries(monkeypatch):
    observed = {}
    timeout = model_service.settings.PREDICTION_PROVIDER_TIMEOUT_SECONDS

    class FakeGenerativeModel:
        def __init__(self, model_name):
            observed["model_name"] = model_name

        def generate_content(self, contents, **kwargs):
            observed["request_options"] = kwargs["request_options"]
            return types.SimpleNamespace(
                text=json.dumps(
                    {
                        "crop": "Apple",
                        "disease": "Apple scab",
                        "confidence": 0.9,
                        "is_healthy": False,
                        "severity": "low",
                    }
                )
            )

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda **kwargs: None
    fake_genai.GenerativeModel = FakeGenerativeModel
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    monkeypatch.setattr(
        "app.services.model_service.settings.GEMINI_API_KEY", "test-only-key"
    )
    monkeypatch.setattr(
        "app.services.model_service.settings.GEMINI_MODEL", "gemini-3.8-flash"
    )

    result = ModelService().predict_gemini(
        Image.new("RGB", (16, 16)),
        image_type="fruit",
        crop_hint="Apple",
    )

    assert observed["model_name"] == "gemini-3.8-flash"
    assert observed["request_options"]["timeout"] == timeout
    assert timeout <= 30
    assert observed["request_options"]["retry"] is None
    assert result["crop"] == "Apple"


def test_openrouter_request_is_bounded_and_does_not_retry_server_errors(monkeypatch):
    import httpx

    requests = []
    timeout = model_service.settings.PREDICTION_PROVIDER_TIMEOUT_SECONDS

    class ServerErrorResponse:
        status_code = 503

    def fake_post(url, **kwargs):
        requests.append(kwargs)
        return ServerErrorResponse()

    monkeypatch.setattr("app.services.model_service.settings.OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setattr(httpx, "post", fake_post)

    result = ModelService().predict_openrouter(
        Image.new("RGB", (16, 16)), image_type="fruit"
    )

    assert result is None
    assert len(requests) == 1
    assert requests[0]["timeout"] == timeout
    assert timeout <= 30


def test_openrouter_retries_with_fallback_model_when_response_is_invalid_json(
    monkeypatch,
):
    import httpx

    responses = [
        type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "choices": [{"message": {"content": "not valid JSON"}}]
                },
            },
        )(),
        type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "model": "meta-llama/llama-3.2-11b-vision-instruct",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "crop": "Banana",
                                        "disease": "Anthracnose",
                                        "confidence": 0.85,
                                        "is_healthy": False,
                                        "severity": "low",
                                    }
                                )
                            }
                        }
                    ],
                },
            },
        )(),
    ]
    requests = []

    def fake_post(url, **kwargs):
        requests.append(kwargs["json"]["model"])
        return responses.pop(0)

    monkeypatch.setattr(
        "app.services.model_service.settings.OPENROUTER_API_KEY", "test-only-key"
    )
    monkeypatch.setattr(httpx, "post", fake_post)

    result = ModelService().predict_openrouter(
        Image.new("RGB", (16, 16)), image_type="fruit"
    )

    assert result["crop"] == "Banana"
    assert len(requests) == 2
    assert (
        requests[0] == "google/gemini-3.8-flash"
        and requests[1] == "meta-llama/llama-3.2-11b-vision-instruct"
    )
