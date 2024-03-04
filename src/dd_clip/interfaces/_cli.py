import json
from pathlib import Path
from typing import Any
from multiprocessing import Manager

from loguru import logger
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dd_clip.core.model import CLIP
from dd_clip.schemas.settings import settings



def load_image(file_path: Path) -> Image.Image:
    try:
        img =  Image.open(file_path).convert("RGB")
        return None if any([x==1 for x in img.size]) else img
    except UnidentifiedImageError:
        logger.error(f"Failed to load image: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load image: {file_path} | {e}")
        return None

def load_json(file_path: Path) -> dict:
    # logger.info(f"Loading json file: {file_path}")
    with open(file_path, "r") as fb:
        return json.load(fb)
    
def write_json(data:dict, file_path:Path):
    with open(file_path, "w") as fb:
        json.dump(data, fb)


def collate(data:list[dict[str, Any]]) -> dict:
    # assumes the same keys are in every file
    data = [x for x in data if x and x["image"] is not None] if data else data
    if data:
        return {k: [dic[k] for dic in data] for k in data[0].keys()}

def uncollate(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Transforms a batched dict into a list of dicts."""
    return [
        {key: value[i] for key, value in data.items()}
        for i in range(len(next(iter(data.values()))))
    ]


class ImageDataset(Dataset):
    def __init__(self, file_paths: list[Path]):
        manager = Manager()
        self.img_paths = manager.list(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx])
        return img

class ReferenceDataset(Dataset):
    def __init__(self, file_paths: list[Path], key: str = "absolute_path"):
        manager = Manager()
        self.file_paths = manager.list(file_paths)
        self.key = key

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = load_json(file_path)
        if data.get("data_type") == "image":
            img_path = data.get(self.key)
            img = load_image(img_path)
            return data | {
                "out_path": str(file_path),
                "image": img,
            }
    

def main(
    in_dir: Path, 
    device: str = settings.device,
    model_dir: Path = settings.model_dir,    
    batch_size: int = settings.max_batch_size, 
    num_workers: int = 4,
    key: str = "absolute_path",
):
    """Go through filles and for image types, embed the image and save the embedding to the file."""

    reference_paths = sorted(list(in_dir.rglob("*.json")))
    dataset = ReferenceDataset(reference_paths, key=key)
    logger.info(f"Found {len(dataset)} reference files.")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
    
    model = CLIP(model_dir=model_dir, device=device)
    
    # loop through dataloader and update progress bar for number of images processed
    n_processed = 0
    pbar = tqdm(total=len(dataloader))
    for i, batch in enumerate(dataloader):
        if batch:
            img_batch: list[Image.Image] = batch.pop("image")
            embeddings = model.embed_imgs(imgs=img_batch, normalized=True)
            batch = uncollate(batch)
            for record, embedding in zip(batch, embeddings):
                out_path = record.pop("out_path")
                record["embedding"] = {"image": embedding.tolist()}
                write_json(record, out_path)
            n_processed += len(img_batch)
            pbar.set_description(f"Batch {i} | Processed {n_processed} images")
        pbar.update(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=settings.device)
    parser.add_argument("--model_dir", type=Path, default=settings.model_dir)
    parser.add_argument("--batch_size", type=int, default=settings.max_batch_size)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--key", type=str, default="absolute_path")
    args = parser.parse_args()
    logger.info(f"Running with args: {args}")
    main(**vars(args))