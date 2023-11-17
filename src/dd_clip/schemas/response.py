from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, conlist


class ModelAnnotation(BaseModel):
    """Generic annotation object"""

    model_name: Optional[str] = Field(None, description="Name of the model that created the annotation")
    model_version: Optional[str] = Field(None, description="Version of the model that created the annotation")

    model_config = ConfigDict(protected_namespaces=())


class ClipResponse(ModelAnnotation):
    embedding: conlist(float, min_length=512, max_length=512) = Field(None, description="CLIP response of embedding")
