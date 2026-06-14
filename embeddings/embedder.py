from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

from config.settings import EMBEDDING_MODEL


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode(self, text: str):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def encode_batch(
        self,
        documents,
        batch_size=128,
        max_length=256
    ):

        # Configure max sequence length directly on the model for SentenceTransformer v3
        self.model.max_seq_length = max_length

        return self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    @property
    def dimension(self):

        return self.model.get_sentence_embedding_dimension()