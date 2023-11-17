import os

from loguru import logger
from pillow_heif import register_heif_opener
import torch
import transformers
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer

register_heif_opener()
transformers.logging.set_verbosity(transformers.logging.CRITICAL)

# when using multithreaded dataloader it breaks things
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIP:
    """Offline CLIP model"""

    def __init__(self, model_dir, device=None, warmup: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model from: {model_dir} on device: {device}")
        
        self.device = device
        self.model_dir = model_dir
        self.model = CLIPModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.img_preprocessor = CLIPImageProcessor()
        if warmup:
            self._warmup()

    def set_device(self, device):
        self.device = device
        self.model = self.model.to(self.device)
        logger.debug(f"Model moved to device: {device}")

    def decomission(self):
        logger.info("Decomissioning model...")
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
        
    def _warmup(self, n: int = 5):
        text = "It's getting hot in here..."
        logger.info(f"Warming up model with {n} inferences...")
        for _ in range(n):
            with torch.inference_mode():
                self.embed_txt(text)
            
            
    def embed_txt(self, text, normalized: bool = True):
        """Embed a single text query"""
        inputs = self.tokenizer(text, 
                                return_tensors='pt',
                                padding=True,
                                truncation=True).to(self.device)
        with torch.inference_mode():
            embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings.cpu().detach().cpu()
        if normalized:
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.numpy()
    
    def embed_imgs(self, imgs, normalized: list[bool] | bool = True):
        """Embed images
        
        Args:
            imgs (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        """
        inputs = self.img_preprocessor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.inference_mode():
            embeddings = self.model.get_image_features(pixel_values=inputs)
        embeddings = embeddings.cpu().detach()
        if normalized and isinstance(normalized, bool):
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        else:
            embeddings = torch.stack([torch.nn.functional.normalize(e, dim=0) if n else e for e, n in zip(embeddings, normalized)])
        return embeddings.numpy()