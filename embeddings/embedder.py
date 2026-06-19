from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

from config.settings import EMBEDDING_MODEL


class Embedder:

    def __init__(self):

        # Load the embedding model once during initialization
        # so it can be reused for all encoding requests.
        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode(self, text: str):

        return self.model.encode(
            text,

            # Normalized embeddings allow cosine similarity
            # to be computed efficiently using dot product.
            normalize_embeddings=True
        )

    def encode_batch(
        self,
        documents,
        batch_size=128,
        max_length=256
    ):

        # Limit token length to control memory usage and
        # maintain consistent embedding generation.
        self.model.max_seq_length = max_length

        return self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    @property
    def dimension(self):

        # Exposes embedding size without requiring callers
        # to access the underlying model directly.
        return self.model.get_sentence_embedding_dimension()