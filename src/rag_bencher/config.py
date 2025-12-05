import os
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str


class RetrieverCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    k: int = Field(4, ge=1, le=100)


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    paths: List[str]


class ProviderModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    region: str | None = None
    chat: Dict[str, Any] | None = None
    embeddings: Dict[str, Any] | None = None


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    offline: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"


class HydeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MultiQueryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    n_queries: int = Field(3, ge=1, le=10)


class RerankCfg(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    method: str = "cosine"
    top_k: int = Field(4, ge=1, le=50)
    cross_encoder_model: Optional[str] = "BAAI/bge-reranker-base"


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: ModelCfg
    retriever: RetrieverCfg
    data: DataCfg
    provider: ProviderModelCfg | None = None
    vector: Dict[str, Any] | None = None
    runtime: RuntimeCfg = RuntimeCfg()
    hyde: HydeCfg | None = None
    multi_query: MultiQueryCfg | None = None
    rerank: RerankCfg | None = None


def load_config(path: str) -> BenchConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = os.path.expandvars(fh.read())
    obj: Dict[str, Any] = yaml.safe_load(raw) or {}
    try:
        return BenchConfig.model_validate(obj)
    except ValidationError as e:
        raise SystemExit(f"Invalid config:\n{e}") from e
