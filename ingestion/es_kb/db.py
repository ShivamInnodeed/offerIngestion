# db.py
"""
SQLAlchemy engine / session helpers with Oracle 19-safe patterns.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker

from .config import AppSettings
from .db_models import Base
from .oracle_types import ORACLE_TS_TZ_FORMAT

_MISSING_DB_URL_MSG = (
    "No database URL configured. Set config db_url, ES_DB_URL, or --db-url."
)


def redact_db_url(db_url: str) -> str:
    """Return a URL safe for logs (password removed)."""
    raw = (db_url or "").strip()
    if not raw:
        return ""
    try:
        # Fast path: no credentials to strip (also preserves sqlite:///... forms).
        if "@" not in raw.split("://", 1)[-1]:
            return raw
        parts = urlsplit(raw)
        if not parts.scheme:
            return raw
        hostname = parts.hostname or ""
        if parts.port:
            netloc = f"{hostname}:{parts.port}"
        else:
            netloc = hostname
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<unparseable-db-url>"


def build_oracle_db_url(
    *,
    user: str,
    password: str,
    host: str,
    port: int | str,
    service_name: str,
) -> str:
    """Construct an Oracle SQLAlchemy URL (password rendered unmasked)."""
    url = URL.create(
        "oracle+oracledb",
        username=user,
        password=password,
        host=host,
        port=int(port),
        query={"service_name": service_name},
    )
    return url.render_as_string(hide_password=False)


def _configure_oracle_session(dbapi_connection, connection_record=None) -> None:
    """Set NLS_TIMESTAMP_TZ_FORMAT on each new Oracle DBAPI connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = "
            f"'{ORACLE_TS_TZ_FORMAT}'"
        )
    finally:
        cursor.close()


def _oracle_json_serializer(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _oracle_json_deserializer(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def create_engine_from_settings(settings: AppSettings) -> Engine:
    """Create a SQLAlchemy Engine from settings.db_url (Oracle or legacy SQLite)."""
    db_url = (settings.db_url or "").strip()
    if not db_url:
        raise ValueError(_MISSING_DB_URL_MSG)

    connect_args: Dict[str, Any] = {}
    engine_kwargs: Dict[str, Any] = {"connect_args": connect_args}

    is_oracle = db_url.startswith("oracle")
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    elif is_oracle:
        engine_kwargs["pool_pre_ping"] = True

    engine = create_engine(db_url, **engine_kwargs)

    if is_oracle:
        engine.dialect._json_serializer = _oracle_json_serializer
        engine.dialect._json_deserializer = _oracle_json_deserializer
        engine.dialect.json_serializer = _oracle_json_serializer
        engine.dialect.json_deserializer = _oracle_json_deserializer
        event.listen(engine, "connect", _configure_oracle_session)

    return engine


def init_db(engine: Engine) -> None:
    """Create missing tables/indexes/constraints. Does not alter existing tables."""
    Base.metadata.create_all(engine)


def verify_oracle_connection(engine: Engine, *, expected_user: str | None = None) -> str:
    """Run SELECT USER FROM dual and optionally assert the connected schema user."""
    with engine.connect() as connection:
        current_user = connection.execute(text("SELECT USER FROM dual")).scalar()
    if current_user is None:
        raise RuntimeError("Oracle connectivity check returned no USER")
    user = str(current_user)
    if expected_user and user.upper() != expected_user.upper():
        raise RuntimeError(f"Expected Oracle user {expected_user!r}, connected as {user!r}")
    return user


def build_session_factory(
    settings: AppSettings,
    *,
    create_schema: bool = False,
) -> sessionmaker[Session]:
    """
    Build a sessionmaker bound to an engine from settings.

    Schema DDL is opt-in via create_schema=True (or init_db / bootstrap script).
    """
    engine = create_engine_from_settings(settings)
    if create_schema:
        init_db(engine)
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
