import os
import torch
import numpy as np


class TextEmbedder:
    """Generate text embeddings using sentence-transformers."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='data/embeddings'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: "
                    "pip install sentence-transformers"
                )
        return self.model
    
    def embed_texts(self, texts, cache_name=None):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            cache_name: Optional name to cache embeddings
            
        Returns:
            torch.Tensor of shape [len(texts), embed_dim]
        """
        if cache_name:
            cache_path = os.path.join(self.cache_dir, f'{cache_name}.pt')
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=False)
        
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        if cache_name:
            torch.save(embeddings, cache_path)
            print(f"Cached embeddings: {cache_path}")
        
        return embeddings
    
    def embed_diseases(self, disease_names, cache_name='disease_embeddings'):
        """
        Generate embeddings for disease names.
        
        Args:
            disease_names: List of disease name strings
            cache_name: Cache file name
            
        Returns:
            torch.Tensor of shape [len(disease_names), embed_dim]
        """
        return self.embed_texts(disease_names, cache_name)
    
    def embed_drugs(self, drug_names, cache_name='drug_embeddings'):
        """
        Generate embeddings for drug names.
        
        Args:
            drug_names: List of drug name strings
            cache_name: Cache file name
            
        Returns:
            torch.Tensor of shape [len(drug_names), embed_dim]
        """
        return self.embed_texts(drug_names, cache_name)
