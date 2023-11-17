from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
from PIL import Image

from dd_clip.core.model import CLIP
from dd_clip.schemas.request import CLIPImageRequest, ClipTextRequest
from dd_clip.schemas.response import ClipResponse
from dd_clip.schemas.settings import settings
from dd_clip.utils.batch import batch
from dd_clip.utils.health import ServiceHealthStatus, service_health
from dd_clip.utils.info import ServiceInfo, service_info
from dd_clip.utils.logging import logger

try:
    import uvicorn
    from fastapi import FastAPI, Form, UploadFile
except ImportError:
    logger.error("Failed to import uvicorn and/or fastapi. Please install them with `pip install uvicorn fastapi`")
    exit(1)


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = CLIP(
        model_dir=settings.model_dir,
        device=settings.device,
    )
    service_health.status = ServiceHealthStatus.OK
    yield
    state["model"].decomission()
    state.clear()


app = FastAPI(title=service_info.title, version=service_info.version, description=service_info.description, lifespan=lifespan)


@batch(max_batch_size=settings.max_batch_size, batch_wait_timeout_s=settings.batch_timeout)
async def handle_img_batch(requests: CLIPImageRequest):
    imgs, normalized = zip(*requests)
    embeddings = state["model"].embed_imgs(imgs=imgs, normalized=normalized)
    return embeddings


@app.get("/health")
async def health() -> ServiceHealthStatus:
    return service_health.status


@app.get("/info")
async def info() -> ServiceInfo:
    return service_info


@app.post("/inference/text", response_model_exclude_none=True)
async def infer_text(request: ClipTextRequest) -> ClipResponse:
    embedding = state["model"].embed_txt(text=request.text, normalized=request.normalized)
    embedding = np.squeeze(embedding)
    return {"embedding": embedding.tolist()}


@app.post("/inference", response_model_exclude_none=True)
async def infer_image(img: UploadFile, normalized: bool = Form(True)) -> ClipResponse:
    img = await img.read()
    img = Image.open(BytesIO(img)).convert("RGB")
    embedding = await handle_img_batch(CLIPImageRequest(img=img, normalized=normalized))
    embedding = np.squeeze(embedding)
    return {"embedding": embedding.tolist()}


def main():
    uvicorn.run("dd_clip.interfaces._api:app", host=settings.host, port=settings.port, reload=settings.reload)


if __name__ == "__main__":
    main()
