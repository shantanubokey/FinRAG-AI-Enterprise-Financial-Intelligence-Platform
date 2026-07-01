"""
JWT and API key authentication.
RBAC: admin can delete documents, user can only query.
"""

from datetime import datetime, timedelta
from typing import Any

from config.logging_config import get_logger

logger = get_logger(__name__)

# In production, store API keys in a database, not in code
_API_KEY_STORE: dict[str, dict[str, Any]] = {}


def create_access_token(
    data: dict[str, Any],
    secret_key: str,
    algorithm: str,
    expires_minutes: int = 60,
) -> str:
    import jwt

    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=expires_minutes)
    payload["iat"] = datetime.utcnow()
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_token(token: str, secret_key: str, algorithm: str) -> dict | None:
    import jwt

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("jwt_token_expired")
        return None
    except jwt.InvalidTokenError:
        return None


def verify_api_key(api_key: str) -> dict | None:
    """Check API key against store. Returns user dict or None."""
    return _API_KEY_STORE.get(api_key)


def register_api_key(api_key: str, user_id: str, role: str = "user") -> None:
    """Register an API key (called at startup or admin setup)."""
    _API_KEY_STORE[api_key] = {"sub": user_id, "role": role}


def require_role(required_role: str):
    """Decorator for role-based access control."""
    from functools import wraps
    from fastapi import HTTPException, status

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user: dict, **kwargs):
            user_role = current_user.get("role", "user")
            role_hierarchy = {"user": 0, "analyst": 1, "admin": 2}
            if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires {required_role} role",
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator
