from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        # 自动用 GPU，没有就回退 CPU
        self.model = SentenceTransformer(model_name, device="cuda")

    def encode(self, texts):
        return self.model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False
        )
    def get_dim(self):
        return self.model.get_sentence_embedding_dimension()