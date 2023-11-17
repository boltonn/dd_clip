from dataclasses import dataclass

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class ClipTextRequest(BaseModel):
    text: str = Field(..., description="Text to embed")
    normalized: bool = Field(True, description="Whether to normalize the embeddings")
    model_config: ConfigDict = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "A dog on a leash which is running.",
                    "normalized": True,
                }
            ]
        }
    }


@dataclass
class CLIPImageRequest:
    img: Image.Image
    normalized: bool = True

    def __iter__(self):
        return iter((self.img, self.normalized))
