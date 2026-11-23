from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import AppException, app_exception_handler
from app.core.security import get_current_user
from app.routers.auth import router as auth_router
from app.services.supabase_service import SupabaseService


class FakeSupabaseService:
    def __init__(self, user=None, auth_enabled=True):
        self.user = user
        self.calls = []
        self.auth_enabled = auth_enabled

    def get_authenticated_user(self, token):
        self.calls.append(token)
        return self.user


def create_auth_test_app(supabase_service):
    app = FastAPI()
    app.state.supabase_service = supabase_service
    app.add_exception_handler(AppException, app_exception_handler)

    @app.get("/private")
    async def private_route(user=Depends(get_current_user)):
        return {"user": user}

    return app


def test_current_user_accepts_verified_supabase_access_token():
    user = {"sub": "supabase-user-id", "email": "farmer@example.com"}
    supabase_service = FakeSupabaseService(user)
    app = create_auth_test_app(supabase_service)

    with TestClient(app) as client:
        response = client.get(
            "/private", headers={"Authorization": "Bearer supabase-access-token"}
        )

    assert response.status_code == 200
    assert response.json()["user"] == user
    assert supabase_service.calls == ["supabase-access-token"]


def test_invalid_bearer_token_is_rejected_not_treated_as_demo_user():
    supabase_service = FakeSupabaseService()
    app = create_auth_test_app(supabase_service)

    with TestClient(app) as client:
        response = client.get(
            "/private", headers={"Authorization": "Bearer expired-token"}
        )

    assert response.status_code == 401
    assert response.json()["error"]["message"] == (
        "Invalid or expired authentication token."
    )


def test_missing_bearer_token_preserves_development_demo_access(monkeypatch):
    monkeypatch.setattr("app.core.security.settings.LOCAL_DEMO_MODE", True)
    app = create_auth_test_app(FakeSupabaseService(auth_enabled=False))

    with TestClient(app) as client:
        response = client.get("/private")

    assert response.status_code == 200
    assert response.json()["user"]["sub"] == "local-demo"


def test_missing_bearer_token_is_rejected_when_supabase_auth_is_configured(
    monkeypatch,
):
    monkeypatch.setattr("app.core.security.settings.LOCAL_DEMO_MODE", True)
    app = create_auth_test_app(FakeSupabaseService(auth_enabled=True))

    with TestClient(app) as client:
        response = client.get("/private")

    assert response.status_code == 401
    assert response.json()["error"]["message"] == (
        "Authentication required. Please log in."
    )


def test_supabase_auth_client_is_initialized_in_demo_mode(monkeypatch):
    fake_client = object()
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(
            LOCAL_DEMO_MODE=True,
            SUPABASE_URL="https://project.supabase.co",
            SUPABASE_ANON_KEY="public-test-key",
        ),
    )
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    service = SupabaseService()

    assert service._auth_client is fake_client
    assert service.is_configured is False


def test_supabase_uses_service_role_for_database_client(monkeypatch):
    settings = SimpleNamespace(
        LOCAL_DEMO_MODE=False,
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_ANON_KEY="public-test-key",
        SUPABASE_SERVICE_ROLE_KEY="server-test-key",
    )
    clients = []

    def fake_create_client(url, key):
        client = object()
        clients.append((url, key, client))
        return client

    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("supabase.create_client", fake_create_client)

    service = SupabaseService()

    assert [client[1] for client in clients] == [
        "public-test-key",
        "server-test-key",
    ]
    assert service._auth_client is clients[0][2]
    assert service._client is clients[1][2]
    assert service.is_configured is True


def test_supabase_persistence_requires_service_role_key(monkeypatch):
    settings = SimpleNamespace(
        LOCAL_DEMO_MODE=False,
        SUPABASE_URL="https://project.supabase.co",
        SUPABASE_ANON_KEY="public-test-key",
        SUPABASE_SERVICE_ROLE_KEY="",
    )
    fake_client = object()
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    service = SupabaseService()

    assert service._auth_client is fake_client
    assert service._client is None
    assert service.is_configured is False
    assert "SUPABASE_SERVICE_ROLE_KEY" in service._persistence_error


def test_supabase_profile_is_built_from_verified_auth_metadata():
    class LocalDemoSupabaseService:
        is_configured = False

        async def get_user_by_id(self, user_id):
            return None

    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.add_exception_handler(AppException, app_exception_handler)
    app.dependency_overrides[get_current_user] = lambda: {
        "sub": "supabase-user-id",
        "email": "farmer@example.com",
        "user_metadata": {"full_name": "A Farmer", "location": "Pune"},
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    }
    app.state.supabase_service = LocalDemoSupabaseService()

    with TestClient(app) as client:
        response = client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "A Farmer"
    assert response.json()["data"]["location"] == "Pune"
    assert response.json()["data"]["id"] == "supabase-user-id"

