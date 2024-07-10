import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_search_engine import BaseSearchEngine
from app_logger import get_logger

logger = get_logger(__name__)

class SearchEngineCLIP(BaseSearchEngine):
    def __init__(self, threshold, **kwargs):
        self._model_id = kwargs.pop("model_id")
        self._device = kwargs.pop("device")
        super().__init__(threshold, **kwargs)
        self._model, self._processor, _ = self._get_model_info(self._model_id, self._device)
        self._embeddings = []
        self._i = 0
        self._size_list = 500
    
    def _get_model_info(self, model_id, device):
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        return model, processor, tokenizer

    def _need_to_add_new_emb(self, new_emb, embeddings, threshold):
        new_emb = self._try_expand_dims(new_emb)
        for emb in embeddings:
            cos_sim = cosine_similarity(new_emb, emb) 
            if cos_sim > threshold:
                return False
        return True
    
    def _try_expand_dims(self, arr):
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def _add_emb(self, emb):
        emb = self._try_expand_dims(emb)
        if len(self._embeddings) == self._size_list:
            self._i %= self._size_list
            self._embeddings[self._i] = emb
        self._embeddings.append(emb)
        self._i += 1

    def run(self, obj):
        img = obj.img
        image = self._processor(
            text = None,
            images = img,
            return_tensors="pt"
            )["pixel_values"].to(self._device)
        embedding = self._model.get_image_features(image)
        # convert the embeddings to numpy array
        embedding_as_np = embedding.cpu().detach().numpy()
        if self._need_to_add_new_emb(embedding_as_np, self._embeddings, self._threshold):
            self._add_emb(embedding_as_np)
            return obj
        return None
        
    
    def run_batch(self, images_with_name):
        images = []
        for img_w_n in images_with_name:
            images.append(img_w_n.img)
        unique_images_w_n = []
        proc_images = self._processor(
                text = None,
                images = images,
                return_tensors="pt"
            )["pixel_values"].to(self._device)
        embeddings = self._model.get_image_features(proc_images)
        # convert the embeddings to numpy array
        embeddings_as_np = embeddings.cpu().detach().numpy()
        for img_w_n, emb in zip(images_with_name, embeddings_as_np):
            if self._need_to_add_new_emb(emb, self._embeddings, self._threshold):
                self._add_emb(emb)
                unique_images_w_n.append(img_w_n)

        return unique_images_w_n
