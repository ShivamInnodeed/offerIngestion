"""FastAPI service for offer ingestion pipeline.

Endpoints:
  GET  /health         – healthcheck
  POST /run-now        – trigger a manual offer ingestion run
  POST /scheduler/start – start recurring schedule (cron or interval)
  POST /scheduler/stop  – stop recurring schedule
  GET  /scheduler/status – show scheduler & run status
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("OFFER_LOG_DIR", str(BASE_DIR / "output" / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_log_fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_fmt)
_fh = logging.FileHandler(str(LOG_DIR / "scheduler_api.log"), encoding="utf-8")
_fh.setFormatter(logging.Formatter(_log_fmt))
logging.getLogger().addHandler(_fh)

logger = logging.getLogger("offer_scheduler_api")

SCRIPT_PATH = BASE_DIR / "offer_ingestion_flow.py"
DEFAULT_OUTPUT_DIR = Path(os.getenv("DEFAULT_OFFER_OUTPUT_DIR", str(BASE_DIR / "output")))
DEFAULT_CONFIG_PATH = os.getenv("DEFAULT_OFFER_CONFIG_PATH", str(BASE_DIR / "offer_config.json"))
DEFAULT_ES_URL = os.getenv("DEFAULT_ES_URL", "http://elasticsearch:9200")


def _normalize_es_url(url: str | None) -> str:
    """Map host-local ES URLs to the in-compose service when running in Docker."""
    resolved = (url or "").strip() or os.getenv("DEFAULT_ES_URL", "http://elasticsearch:9200")
    docker_es = os.getenv("DEFAULT_ES_URL", "").strip()
    if docker_es and ("localhost" in resolved or "127.0.0.1" in resolved):
        return docker_es
    return resolved


SCHEDULER_API_KEY = os.getenv("SCHEDULER_API_KEY", "")
SCHEDULER_JOB_ID = "offer_ingestion_recurring"


class RunOptions(BaseModel):
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    config_path: str = DEFAULT_CONFIG_PATH
    api_url: str | None = Field(default=None, description="REST API URL to fetch offer XML")
    xml_input: str | None = Field(default=None, description="Path to local XML file")
    run_fetch_script: bool = Field(
        default=False,
        description="If true, run offerIngestion/script.py to fetch fresh XML before parsing.",
    )
    skip_fetch: bool = Field(
        default=False,
        description="Skip XML fetch. Implied when index_from_updated_offers is true.",
    )
    index_from_updated_offers: bool = Field(
        default=False,
        description="Index from output/updated_offers.json (skip fetch, XML parse, LLM, change detection).",
    )
    updated_offers_file: str | None = Field(
        default=None,
        description="Explicit path to updated_offers.json (default: output_dir/updated_offers.json).",
    )
    index_name: str | None = Field(
        default=None,
        description="Elasticsearch index name override (e.g. kb_documents_offers).",
    )
    es_url: str = DEFAULT_ES_URL
    es_username: str | None = None
    es_password: str | None = None
    db_url: str | None = None
    model_name: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)
    recreate_index: bool = False
    enable_llm_enrichment: bool = False
    max_llm_offers: int | None = Field(default=None, ge=1)
    state_output: str | None = None

    @model_validator(mode="after")
    def fill_derived(self) -> "RunOptions":
        output_dir = Path(self.output_dir)
        if not self.state_output:
            self.state_output = str(output_dir / "offer_ingestion_state.json")
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 6)
        if self.index_from_updated_offers:
            self.skip_fetch = True
            self.run_fetch_script = False
            self.enable_llm_enrichment = False
        self.es_url = _normalize_es_url(self.es_url)
        return self


class ScheduleStartRequest(BaseModel):
    interval_seconds: int | None = Field(default=None, ge=5)
    cron: str | None = None
    run_options: RunOptions = Field(default_factory=RunOptions)

    @model_validator(mode="after")
    def validate_schedule(self) -> "ScheduleStartRequest":
        if self.interval_seconds is None and not (self.cron and self.cron.strip()):
            raise ValueError("Provide either interval_seconds or cron")
        if self.interval_seconds is not None and self.cron:
            raise ValueError("Use only one of interval_seconds or cron")
        return self


@dataclass
class ActiveRun:
    run_id: str
    started_at: str
    trigger: str
    command: list[str]
    log_path: str
    process: subprocess.Popen[Any]
    log_file: Any


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: ActiveRun | None = None
        self._last_result: dict[str, Any] | None = None

    def start_run(self, options: RunOptions, trigger: str) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("An offer ingestion run is already in progress")

            run_id = str(uuid.uuid4())
            output_dir = Path(options.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f"offer_run_{run_id}.log"
            command = self._build_command(options)
            env = os.environ.copy()
            ingestion_dir = env.get("INGESTION_DIR", "/app/ingestion")
            env["PYTHONPATH"] = f"{BASE_DIR}:{ingestion_dir}"
            env.setdefault("INGESTION_DIR", ingestion_dir)
            env.setdefault("DEFAULT_ES_URL", DEFAULT_ES_URL)
            env.setdefault("ES_ELASTICSEARCH_URL", DEFAULT_ES_URL)

            log_file = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active = ActiveRun(
                run_id=run_id,
                started_at=_utc_now(),
                trigger=trigger,
                command=command,
                log_path=str(log_path),
                process=process,
                log_file=log_file,
            )
            self._active = active
            self._last_result = None
            threading.Thread(target=self._watch_process, args=(active,), daemon=True).start()

            return {
                "status": "started",
                "run_id": run_id,
                "started_at": active.started_at,
                "trigger": trigger,
                "pid": process.pid,
                "log_path": str(log_path),
                "command": " ".join(shlex.quote(part) for part in command),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            active_payload: dict[str, Any] | None = None
            if active is not None:
                active_payload = {
                    "run_id": active.run_id,
                    "started_at": active.started_at,
                    "trigger": active.trigger,
                    "pid": active.process.pid,
                    "log_path": active.log_path,
                }
            return {
                "running": active is not None,
                "active_run": active_payload,
                "last_run": self._last_result,
            }

    def _watch_process(self, run: ActiveRun) -> None:
        exit_code = run.process.wait()
        run.log_file.flush()
        run.log_file.close()
        result = {
            "run_id": run.run_id,
            "trigger": run.trigger,
            "started_at": run.started_at,
            "completed_at": _utc_now(),
            "exit_code": exit_code,
            "status": "success" if exit_code == 0 else "failed",
            "log_path": run.log_path,
        }
        with self._lock:
            if self._active and self._active.run_id == run.run_id:
                self._active = None
            self._last_result = result
        logger.info("Offer run completed run_id=%s exit_code=%s", run.run_id, exit_code)

    @staticmethod
    def _build_command(options: RunOptions) -> list[str]:
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--output-dir", options.output_dir,
            "--config", options.config_path,
            "--es-url", options.es_url,
            "--chunk-size", str(options.chunk_size),
            "--chunk-overlap", str(options.chunk_overlap),
        ]
        if options.run_fetch_script:
            command.append("--run-fetch-script")
        if options.skip_fetch:
            command.append("--skip-fetch")
        if options.index_from_updated_offers:
            command.append("--index-from-updated-offers")
        if options.updated_offers_file:
            command.extend(["--updated-offers-file", options.updated_offers_file])
        if options.index_name:
            command.extend(["--index-name", options.index_name])
        if options.api_url:
            command.extend(["--api-url", options.api_url])
        if options.xml_input:
            command.extend(["--xml-input", options.xml_input])
        if options.model_name:
            command.extend(["--model-name", options.model_name])
        if options.batch_size is not None:
            command.extend(["--batch-size", str(options.batch_size)])
        if options.es_username:
            command.extend(["--es-username", options.es_username])
        if options.es_password:
            command.extend(["--es-password", options.es_password])
        if options.db_url:
            command.extend(["--db-url", options.db_url])
        if options.recreate_index:
            command.append("--recreate-index")
        if options.enable_llm_enrichment:
            command.append("--enable-llm-enrichment")
        if options.enable_llm_enrichment and options.max_llm_offers:
            command.extend(["--max-llm-offers", str(options.max_llm_offers)])
        if options.state_output:
            command.extend(["--output", options.state_output])
        return command


class SchedulerManager:
    def __init__(self, run_manager: RunManager) -> None:
        self._run_manager = run_manager
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduler.start()
        self._lock = threading.Lock()
        self._schedule_config: dict[str, Any] | None = None

    def start(self, request: ScheduleStartRequest) -> dict[str, Any]:
        with self._lock:
            if self._scheduler.get_job(SCHEDULER_JOB_ID) is not None:
                raise RuntimeError("Scheduler is already running")

            if request.interval_seconds is not None:
                self._scheduler.add_job(
                    self._scheduled_run, "interval",
                    seconds=request.interval_seconds,
                    id=SCHEDULER_JOB_ID, replace_existing=False,
                )
                self._schedule_config = {
                    "mode": "interval",
                    "interval_seconds": request.interval_seconds,
                    "run_options": request.run_options.model_dump(),
                }
            else:
                parts = request.cron.strip().split()
                if len(parts) != 5:
                    raise ValueError("cron must be standard 5-part: m h dom mon dow")
                minute, hour, day, month, dow = parts
                self._scheduler.add_job(
                    self._scheduled_run,
                    trigger=CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=dow, timezone="UTC"),
                    id=SCHEDULER_JOB_ID, replace_existing=False,
                )
                self._schedule_config = {
                    "mode": "cron",
                    "cron": request.cron,
                    "run_options": request.run_options.model_dump(),
                }

            self._schedule_config["started_at"] = _utc_now()
            self._schedule_config["next_run_at"] = self._next_run_time()
            return {"status": "started", "schedule": self._schedule_config}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._scheduler.get_job(SCHEDULER_JOB_ID) is None:
                return {"status": "not_running"}
            self._scheduler.remove_job(SCHEDULER_JOB_ID)
            stopped = {"status": "stopped", "stopped_at": _utc_now(), "previous_schedule": self._schedule_config}
            self._schedule_config = None
            return stopped

    def status(self) -> dict[str, Any]:
        with self._lock:
            job = self._scheduler.get_job(SCHEDULER_JOB_ID)
            return {
                "running": job is not None,
                "next_run_at": self._next_run_time(),
                "schedule": self._schedule_config,
            }

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)

    def _scheduled_run(self) -> None:
        with self._lock:
            schedule = self._schedule_config or {}
            options_data = schedule.get("run_options") or RunOptions().model_dump()
        options = RunOptions.model_validate(options_data)
        try:
            self._run_manager.start_run(options, trigger="scheduler")
        except RuntimeError:
            logger.warning("Scheduled run skipped (another run in progress)")
        except Exception:
            logger.exception("Scheduled run failed to start")

    def _next_run_time(self) -> str | None:
        job = self._scheduler.get_job(SCHEDULER_JOB_ID)
        if not job or not job.next_run_time:
            return None
        return job.next_run_time.astimezone(timezone.utc).isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not SCHEDULER_API_KEY:
        raise HTTPException(status_code=500, detail="SCHEDULER_API_KEY is not configured")
    if x_api_key != SCHEDULER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Offer Ingestion Scheduler API", version="1.0.0")
run_manager = RunManager()
scheduler_manager = SchedulerManager(run_manager=run_manager)


@app.on_event("shutdown")
def _on_shutdown() -> None:
    scheduler_manager.shutdown()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-now")
def run_now(request: RunOptions, _: None = Depends(_require_api_key)) -> dict[str, Any]:
    try:
        return run_manager.start_run(request, trigger="manual")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/scheduler/start")
def scheduler_start(request: ScheduleStartRequest, _: None = Depends(_require_api_key)) -> dict[str, Any]:
    try:
        return scheduler_manager.start(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/scheduler/stop")
def scheduler_stop(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    return scheduler_manager.stop()


@app.get("/scheduler/status")
def scheduler_status(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    return {
        "scheduler": scheduler_manager.status(),
        "runs": run_manager.status(),
    }


@app.get("/run-status")
def run_status() -> dict[str, Any]:
    return run_manager.status()


@app.get("/logs")
def list_logs() -> dict[str, Any]:
    """List available log files."""
    log_files = []
    for p in sorted(LOG_DIR.glob("*.log"), reverse=True):
        log_files.append({"name": p.name, "size_bytes": p.stat().st_size})
    output_log_dir = DEFAULT_OUTPUT_DIR
    for p in sorted(output_log_dir.glob("offer_run_*.log"), reverse=True):
        log_files.append({"name": p.name, "size_bytes": p.stat().st_size})
    return {"log_dir": str(LOG_DIR), "files": log_files}


@app.get("/logs/{filename}")
def read_log(filename: str, tail: int = 200) -> dict[str, Any]:
    """Read last N lines of a log file."""
    log_path = LOG_DIR / filename
    if not log_path.is_file():
        log_path = DEFAULT_OUTPUT_DIR / filename
    if not log_path.is_file():
        raise HTTPException(status_code=404, detail=f"Log file not found: {filename}")
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"filename": filename, "total_lines": len(lines), "lines": lines[-tail:]}
