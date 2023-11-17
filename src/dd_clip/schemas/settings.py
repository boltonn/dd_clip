from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dd_clip.schemas.enums.ctranslate2 import Ctranslate2ComputeType


class Settings(BaseSettings):
    host: str = Field("0.0.0.0", validation_alias="HOST", description="Host to bind to")
    port: int = Field(8080, validation_alias="PORT", description="Port to bind to")
    model_dir: Path = Field(..., description="Path to HuggingFace CLIP model directory")
    max_batch_size: int = Field(1, description="Maximum batch size for batching")
    batch_timeout: float = Field(0, description="Timeout for batching in seconds")
    device: Optional[str] = Field(None, description="Device to use for the model")
    compute_type: Optional[Ctranslate2ComputeType] = Field("int8", description="Compute type to use for the model")
    reload: Optional[bool] = Field(True, description="Reload mode")
    log_file: Optional[str] = Field(None, description="File to write logs to if desired")
    log_level: Optional[str] = Field("INFO", validation_alias="LOG_LEVEL", description="Log level")
    model_config = SettingsConfigDict(env_file=".env", protected_namespaces=[])


settings = Settings()
