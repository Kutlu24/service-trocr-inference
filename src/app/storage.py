"""
Storage layer for all job statuses.

Supports:
- inference
- xml_merge
- evaluation

Backends:
1. SQLModel (SQLite / Postgres)
2. JSON file

Exports:
- inference-status.json
- xml-merge-status.json
- evaluation-status.json
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Type

import aiofiles
from loguru import logger
from sqlmodel import SQLModel, Field, select, col
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from app.models.inference_models import InferenceResponseModel
from app.models.raw_xml_upload_models import RawXMLMergeResponseModel
from app.models.evaluation_models import EvaluationResponseModel
from pydantic import BaseModel


# -----------------------------
# Registry of job models
# -----------------------------

JOB_MODELS: dict[str, Type[BaseModel]] = {
    "inference": InferenceResponseModel,
    "xml_merge": RawXMLMergeResponseModel,
    "evaluation": EvaluationResponseModel,
}


# -----------------------------
# SQLModel Table
# -----------------------------

class JobStatusDB(SQLModel, table=True):
    __tablename__ = "job_status"

    request_id: str = Field(primary_key=True, index=True)
    job_type: str = Field(index=True)
    source: str = Field(index=True)

    state: str = Field(index=True)
    created_at: str = Field(index=True)
    runtime_seconds: float = Field(default=0.0)

    payload_json: str

    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# -----------------------------
# Abstract repository
# -----------------------------

class StatusRepository(ABC):
    @abstractmethod
    async def save(self, status) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, request_id: str):
        pass

    @abstractmethod
    async def get_all(self, job_type: Optional[str] = None):
        pass

    @abstractmethod
    async def export_to_json(self, output_path: Path, job_type: Optional[str] = None):
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


# -----------------------------
# SQLModel repository
# -----------------------------

class SQLModelStatusRepository(StatusRepository):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

        url = f"sqlite+aiosqlite:///{db_path}"
        self.engine = create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        logger.info(f"SQLModel engine created for {db_path}")

    async def _init_db(self):
        if self._initialized:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    @staticmethod
    def _pydantic_to_db(status) -> JobStatusDB:
        return JobStatusDB(
            request_id=status.request_id,
            job_type=status.job_type,
            source=status.source,
            state=status.state.value,
            created_at=status.created_at.isoformat(),
            runtime_seconds=status.runtime_seconds,
            payload_json=status.model_dump_json(by_alias=True),
            updated_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _db_to_pydantic(db_row: JobStatusDB):
        model = JOB_MODELS.get(db_row.job_type)
        if not model:
            raise ValueError(f"Unknown job_type in DB: {db_row.job_type}")
        return model.model_validate(json.loads(db_row.payload_json))

    async def save(self, status):
        await self._init_db()

        db_status = self._pydantic_to_db(status)

        async with AsyncSession(self.engine) as session:
            existing = await session.get(JobStatusDB, status.request_id)

            if existing:
                for key, value in db_status.model_dump().items():
                    setattr(existing, key, value)
                db_status = existing

            session.add(db_status)
            await session.commit()

        logger.info(
            "Saved job status | request_id=%s | job_type=%s | state=%s",
            status.request_id,
            status.job_type,
            status.state.value,
        )

    async def get_by_id(self, request_id: str):
        await self._init_db()
        async with AsyncSession(self.engine) as session:
            row = await session.get(JobStatusDB, request_id)
            if not row:
                return None
            return self._db_to_pydantic(row)

    async def get_all(self, job_type: Optional[str] = None):
        await self._init_db()
        async with AsyncSession(self.engine) as session:
            stmt = select(JobStatusDB)
            if job_type:
                stmt = stmt.where(JobStatusDB.job_type == job_type)
            stmt = stmt.order_by(col(JobStatusDB.created_at).desc())
            rows = (await session.execute(stmt)).scalars().all()

            return [self._db_to_pydantic(row) for row in rows]

    async def export_to_json(self, output_path: Path, job_type: Optional[str] = None):
        data = await self.get_all(job_type)
        payload = [s.model_dump(by_alias=True, mode="json") for s in data]
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    async def close(self):
        await self.engine.dispose()
        logger.info(f"SQLModel engine disposed for {self.db_path}")


# -----------------------------
# JSON repository
# -----------------------------

class JSONStatusRepository(StatusRepository):
    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            path.write_text("[]", encoding="utf-8")
            logger.info(f"Created new JSON file at {self.path}")

    async def _read(self):
        async with aiofiles.open(self.path, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def _write(self, data):
        async with aiofiles.open(self.path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, ensure_ascii=False))

    async def save(self, status):
        data = await self._read()

        # Use JSON-safe Pydantic serialization, just like SQL backend
        payload = json.loads(status.model_dump_json(by_alias=True))

        data = [d for d in data if d["request_id"] != status.request_id]
        data.append(payload)

        await self._write(data)

        logger.info(
            "Saved job status | request_id=%s | job_type=%s | state=%s",
            status.request_id,
            status.job_type,
            status.state.value,
        )

    async def get_by_id(self, request_id: str):
        for row in await self._read():
            if row["request_id"] == request_id:
                model = JOB_MODELS.get(row["job_type"])
                if not model:
                    raise ValueError(f"Unknown job_type in JSON: {row['job_type']}")
                return model(**row)
        return None

    async def get_all(self, job_type: Optional[str] = None):
        rows = await self._read()

        # Match SQL ordering: newest first
        rows = sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)

        if job_type:
            rows = [r for r in rows if r["job_type"] == job_type]

        result = []
        for row in rows:
            model = JOB_MODELS.get(row["job_type"])
            if not model:
                raise ValueError(f"Unknown job_type in JSON: {row['job_type']}")
            result.append(model(**row))
        return result

    async def export_to_json(self, output_path: Path, job_type: Optional[str] = None):
        data = await self.get_all(job_type)
        payload = [s.model_dump(by_alias=True, mode="json") for s in data]

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    async def close(self):
        logger.info(f"Closing JSON repository at {self.path}")


# -----------------------------
# Factory
# -----------------------------

def create_repository(storage_type: str = "sqlite", path: Path | None = None) -> StatusRepository:
    if path is None:
        path = Path("job-status.db" if storage_type == "sqlite" else "job-status.json")

    if storage_type == "sqlite":
        return SQLModelStatusRepository(path)
    elif storage_type == "json":
        return JSONStatusRepository(path)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
