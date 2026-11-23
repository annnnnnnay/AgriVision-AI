"""
Supabase service — handles database operations for users, predictions, and storage.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SupabaseService:
    """Centralized Supabase client for database, auth, and storage operations."""

    def __init__(self):
        self._client = None
        self._auth_client = None
        self._is_configured = False
        self._persistence_error: Optional[str] = None
        self._local_predictions: List[Dict[str, Any]] = []
        self._local_users: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Supabase client from environment variables."""
        from app.config import get_settings
        settings = get_settings()

        supabase_url = getattr(settings, "SUPABASE_URL", "") or ""
        supabase_key = getattr(settings, "SUPABASE_ANON_KEY", "") or ""
        service_role_key = getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or ""

        if not supabase_url or not supabase_key:
            logger.warning(
                "Supabase not configured (SUPABASE_URL/SUPABASE_ANON_KEY missing). "
                "Running in local-only mode — data will NOT be persisted."
            )
            return

        try:
            from supabase import create_client, Client
            self._auth_client: Client = create_client(supabase_url, supabase_key)
            if settings.LOCAL_DEMO_MODE:
                logger.info(
                    "Local demo mode enabled; Supabase Auth token validation is available, "
                    "but database persistence is disabled."
                )
                return

            if not service_role_key:
                self._persistence_error = (
                    "SUPABASE_SERVICE_ROLE_KEY is required for server-side "
                    "prediction history and image storage."
                )
                logger.warning(self._persistence_error)
                return

            self._client = create_client(supabase_url, service_role_key)
            self._is_configured = True
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")

    def _ensure_persistence_configured(self) -> None:
        if self._persistence_error:
            raise RuntimeError(self._persistence_error)

    def get_authenticated_user(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Validate a Supabase access token and return its user identity."""
        if self._auth_client is None:
            return None

        try:
            response = self._auth_client.auth.get_user(access_token)
            user = response.user
            if user is None:
                return None

            metadata = getattr(user, "user_metadata", None) or {}
            return {
                "sub": str(user.id),
                "email": user.email,
                "user_metadata": metadata,
                "created_at": (
                    user.created_at.isoformat()
                    if hasattr(user.created_at, "isoformat")
                    else user.created_at
                ),
            }
        except Exception as e:
            logger.info("Supabase access-token validation failed: %s", e)
            return None

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    @property
    def auth_enabled(self) -> bool:
        return self._auth_client is not None

    # ============================================================
    # PREDICTION HISTORY
    # ============================================================

    async def save_prediction(
        self,
        user_id: Optional[str],
        prediction_data: Dict[str, Any],
        image_url: Optional[str] = None,
    ) -> Optional[Dict]:
        """Save a prediction result to the database."""
        if not self._is_configured:
            self._ensure_persistence_configured()
            record = {
                "id": str(uuid4()),
                "user_id": user_id,
                "disease": prediction_data.get("disease"),
                "disease_id": prediction_data.get("disease_id"),
                "crop": prediction_data.get("crop"),
                "confidence": prediction_data.get("confidence"),
                "severity": prediction_data.get("severity"),
                "is_healthy": prediction_data.get("is_healthy"),
                "language": prediction_data.get("language", "en"),
                "processing_time_ms": prediction_data.get("processing_time_ms"),
                "image_url": image_url,
                "top_predictions": prediction_data.get("top_predictions"),
                "disease_info": prediction_data.get("disease_info"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._local_predictions.insert(0, record)
            return record

        try:
            record = {
                "id": str(uuid4()),
                "user_id": user_id,
                "disease": prediction_data.get("disease"),
                "disease_id": prediction_data.get("disease_id"),
                "crop": prediction_data.get("crop"),
                "confidence": prediction_data.get("confidence"),
                "severity": prediction_data.get("severity"),
                "is_healthy": prediction_data.get("is_healthy"),
                "language": prediction_data.get("language", "en"),
                "processing_time_ms": prediction_data.get("processing_time_ms"),
                "image_url": image_url,
                "top_predictions": prediction_data.get("top_predictions"),
                "disease_info": prediction_data.get("disease_info"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            result = self._client.table("predictions").insert(record).execute()
            logger.info(f"Prediction saved: {record['id']}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return None

    async def get_prediction_history(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict:
        """Get prediction history for a user."""
        if not self._is_configured:
            self._ensure_persistence_configured()
            predictions = [
                record for record in self._local_predictions
                if record.get("user_id") == user_id
            ]
            return {
                "predictions": predictions[offset:offset + limit],
                "total": len(predictions),
                "has_more": offset + limit < len(predictions),
            }

        try:
            # Get total count
            count_result = (
                self._client.table("predictions")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .execute()
            )
            total = count_result.count or 0

            # Get paginated results
            result = (
                self._client.table("predictions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            return {
                "predictions": result.data or [],
                "total": total,
                "has_more": (offset + limit) < total,
            }
        except Exception as e:
            logger.exception("Failed to get prediction history: %s", e)
            raise

    # ============================================================
    # ANALYTICS
    # ============================================================

    async def get_user_analytics(self, user_id: str) -> Dict:
        """Get analytics data for a user."""
        if not self._is_configured:
            self._ensure_persistence_configured()

        try:
            if self._is_configured:
                result = (
                    self._client.table("predictions")
                    .select("disease, crop, severity, is_healthy, created_at")
                    .eq("user_id", user_id)
                    .execute()
                )
                predictions = result.data or []
            else:
                predictions = [
                    record for record in self._local_predictions
                    if record.get("user_id") == user_id
                ]
            if not predictions:
                return self._empty_analytics()

            total = len(predictions)
            healthy_count = sum(1 for p in predictions if p.get("is_healthy"))
            diseased_count = total - healthy_count

            # Disease distribution
            disease_counts: Dict[str, int] = {}
            crop_counts: Dict[str, int] = {}
            severity_dist = {"high": 0, "moderate": 0, "low": 0}

            for p in predictions:
                d = p.get("disease", "Unknown")
                c = p.get("crop", "Unknown")
                s = p.get("severity")

                disease_counts[d] = disease_counts.get(d, 0) + 1
                crop_counts[c] = crop_counts.get(c, 0) + 1
                if s and s in severity_dist:
                    severity_dist[s] += 1

            # Most common
            most_common_disease = max(disease_counts, key=disease_counts.get) if disease_counts else None
            most_common_crop = max(crop_counts, key=crop_counts.get) if crop_counts else None

            # This month
            now = datetime.now(timezone.utc)
            predictions_this_month = sum(
                1 for p in predictions
                if p.get("created_at", "").startswith(now.strftime("%Y-%m"))
            )

            return {
                "total_predictions": total,
                "healthy_count": healthy_count,
                "diseased_count": diseased_count,
                "most_common_disease": most_common_disease,
                "most_common_crop": most_common_crop,
                "disease_distribution": [
                    {"name": k, "count": v}
                    for k, v in sorted(disease_counts.items(), key=lambda x: -x[1])[:10]
                ],
                "crop_distribution": [
                    {"name": k, "count": v}
                    for k, v in sorted(crop_counts.items(), key=lambda x: -x[1])
                ],
                "severity_distribution": severity_dist,
                "predictions_this_month": predictions_this_month,
            }
        except Exception as e:
            logger.exception("Failed to get analytics: %s", e)
            raise

    @staticmethod
    def _empty_analytics() -> Dict:
        return {
            "total_predictions": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "most_common_disease": None,
            "most_common_crop": None,
            "disease_distribution": [],
            "crop_distribution": [],
            "severity_distribution": {"high": 0, "moderate": 0, "low": 0},
            "predictions_this_month": 0,
        }

    # ============================================================
    # IMAGE STORAGE
    # ============================================================

    async def upload_image(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Upload an image to Supabase Storage and return the public URL."""
        if not self._is_configured:
            return None

        try:
            bucket = "crop-images"
            path = f"uploads/{uuid4().hex[:8]}_{filename}"

            self._client.storage.from_(bucket).upload(
                path, file_bytes, {"content-type": "image/jpeg"}
            )

            url = self._client.storage.from_(bucket).get_public_url(path)
            return url
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            return None

    # ============================================================
    # USER PROFILES
    # ============================================================

    async def create_user(
        self,
        name: str,
        email: str,
        password_hash: str,
        location: Optional[str] = None,
    ) -> Optional[Dict]:
        """Create a new user profile record."""
        if not self._is_configured:
            record = {
                "id": str(uuid4())[:12],
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "location": location,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._local_users[record["id"]] = record
            return record

        try:
            record = {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "location": location,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            result = self._client.table("user_profiles").insert(record).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve user profile by email."""
        if not self._is_configured:
            return next(
                (user for user in self._local_users.values() if user["email"].lower() == email.lower()),
                None,
            )

        try:
            result = (
                self._client.table("user_profiles")
                .select("*")
                .eq("email", email)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Retrieve user profile by ID."""
        if not self._is_configured:
            return self._local_users.get(user_id)

        try:
            result = (
                self._client.table("user_profiles")
                .select("*")
                .eq("id", user_id)
                .execute()
            )
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
