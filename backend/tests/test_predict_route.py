from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.core.exceptions import AppException, app_exception_handler
from app.core.security import get_current_user
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
    app.add_exception_handler(AppException, app_exception_handler)
    app.dependency_overrides[get_current_user] = lambda: {}
    app.state.model_service = DecliningModelService()

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
