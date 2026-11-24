from io import BytesIO
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.core.exceptions import AppException, app_exception_handler
from app.core.security import get_current_user_optional
from app.rate_limit import limiter, prediction_rate_limit
from app.routers.predict import router


class DecliningModelService:
    def predict(self, image, language, image_type, crop_hint):
        raise RuntimeError(
            "The local model supports leaf photos only; configure a vision provider "
            "for fruit, other, or unspecified images."
        )


def test_fruit_image_without_cloud_provider_returns_scope_error():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.limiter = limiter
    app.add_exception_handler(AppException, app_exception_handler)
    app.dependency_overrides[get_current_user_optional] = lambda: {}
    app.state.model_service = DecliningModelService()
    app.state.supabase_service = SimpleNamespace(auth_enabled=True)

    image_bytes = BytesIO()
    Image.new("RGB", (16, 16)).save(image_bytes, format="PNG")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("apple.png", image_bytes.getvalue(), "image/png")},
            data={"image_type": "fruit", "crop_hint": "Apple"},
        )

    assert response.status_code == 422
    assert "supports leaf photos only" in response.json()["error"]["message"]


def test_guest_prediction_is_rejected_when_guest_mode_is_disabled(monkeypatch):
    monkeypatch.setattr(
        "app.routers.predict.settings.ENABLE_GUEST_ANALYSIS", False
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.limiter = limiter
    app.add_exception_handler(AppException, app_exception_handler)
    app.state.supabase_service = SimpleNamespace(auth_enabled=True)
    app.state.model_service = DecliningModelService()

    image_bytes = BytesIO()
    Image.new("RGB", (16, 16)).save(image_bytes, format="PNG")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", image_bytes.getvalue(), "image/png")},
            data={"image_type": "leaf"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == (
        "Sign in is required to analyze crop images."
    )


def test_guest_prediction_runs_without_persisting_history(monkeypatch):
    class SuccessfulModelService:
        def predict(self, image, language, image_type, crop_hint):
            return {
                "predicted_class": "Tomato___healthy",
                "confidence": 0.9,
                "crop": "Tomato",
                "severity": None,
                "is_healthy": True,
                "top_predictions": [
                    {"class": "Tomato___healthy", "confidence": 0.9}
                ],
                "processing_time_ms": 12,
                "model_version": "test-model",
                "gemini_rich_info": {
                    "name": "Healthy",
                    "crop": "Tomato",
                    "description": "Healthy leaf.",
                    "symptoms": [],
                    "causes": [],
                    "treatment": {"organic": [], "chemical": []},
                    "prevention": [],
                },
            }

    monkeypatch.setattr("app.routers.predict.settings.ENABLE_GUEST_ANALYSIS", True)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.limiter = limiter
    app.add_exception_handler(AppException, app_exception_handler)
    app.state.supabase_service = SimpleNamespace(
        auth_enabled=True,
        upload_image=lambda *args: pytest.fail("Guest uploads must not be stored."),
        save_prediction=lambda **kwargs: pytest.fail(
            "Guest prediction history must not be saved."
        ),
    )
    app.state.model_service = SuccessfulModelService()
    app.state.disease_service = object()
    app.state.translation_service = object()

    image_bytes = BytesIO()
    Image.new("RGB", (16, 16)).save(image_bytes, format="PNG")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", image_bytes.getvalue(), "image/png")},
            data={"image_type": "leaf"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["crop"] == "Tomato"


def test_guest_and_authenticated_prediction_limits_are_distinct():
    assert prediction_rate_limit("guest:203.0.113.1") == "3/hour"
    assert prediction_rate_limit("authenticated:203.0.113.1") == "30/minute"
