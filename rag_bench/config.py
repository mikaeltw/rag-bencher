
from __future__ import annotations
from typing import List
import os
import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict

class ModelCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(..., description="LLM model id (e.g., gpt-4o-mini)")

class RetrieverCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    k: int = Field(4, ge=1, le=100, description="Top-K documents to retrieve")

class DataCfg(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    paths: List[str] = Field(..., min_length=1, description="List of text file paths")

class BenchConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model: ModelCfg
    retriever: RetrieverCfg
    data: DataCfg

def _expand_env(text: str) -> str:
    # Expand ${VAR} and $VAR using current environment
    return os.path.expandvars(text)

def load_config(path: str) -> BenchConfig:
    # Read raw YAML, expand env vars, then parse + validate strictly
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    expanded = _expand_env(raw)
    obj = yaml.safe_load(expanded) or {}
    try:
        return BenchConfig.model_validate(obj)
    except ValidationError as e:
        # Pretty error for CLI users
        raise SystemExit(f"Invalid config:\n{e}")
