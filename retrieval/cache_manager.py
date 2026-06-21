import json
import pickle
from pathlib import Path

import faiss # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]


# Handles all persistence of precomputed artifacts to disk.
# Separates cache I/O from the embedding pipeline so the pipeline
# can focus purely on computation without managing file paths.
# All four artifacts must be present together for a valid cache hit.
class CacheManager:

    def __init__(self):

        # Single cache directory keeps all artifacts co-located
        # and simplifies cache invalidation — delete the folder to reset.
        self.cache_dir = Path("cache")

        self.cache_dir.mkdir(
            exist_ok=True
        )

        # Raw candidate embeddings stored as a NumPy array.
        # Loaded directly into memory for FAISS index reconstruction
        # or fallback exhaustive search if needed.
        self.embedding_file = (
            self.cache_dir /
            "candidate_embeddings.npy"
        )

        # Ordered list of candidate IDs stored in the same sequence
        # as the embedding rows, enabling index-to-ID resolution
        # after FAISS returns positional results.
        self.id_file = (
            self.cache_dir /
            "candidate_ids.pkl"
        )

        # Serialized FAISS index enabling sub-second approximate
        # nearest-neighbor search over large candidate embedding sets.
        self.faiss_file = (
            self.cache_dir /
            "faiss.index"
        )

        # JSON metadata stores dataset statistics and model provenance.
        # Used to detect stale caches when the source data or model changes.
        self.metadata_file = (
            self.cache_dir /
            "metadata.json"
        )

    def cache_exists(self):

        # All four artifacts must exist together for the cache to be valid.
        # A partial cache — e.g. missing the FAISS index — would cause
        # silent failures during retrieval and is treated as a cache miss.
        return (

            self.embedding_file.exists()

            and

            self.id_file.exists()

            and

            self.faiss_file.exists()

            and

            self.metadata_file.exists()

        )

    def save(

        self,

        embeddings,

        ids,

        index,

        metadata

    ):

        # NumPy's native .npy format preserves array dtype and shape
        # exactly, avoiding precision loss from text-based serialization.
        np.save(

            self.embedding_file,

            embeddings

        )

        # Pickle is used for the ID list because it handles arbitrary
        # Python objects and maintains insertion order reliably.
        with open(

            self.id_file,

            "wb"

        ) as file:

            pickle.dump(

                ids,

                file

            )

        # FAISS provides its own optimized binary serialization format
        # that preserves the index structure and trained quantizer state.
        faiss.write_index(

            index,

            str(self.faiss_file)

        )

        # Metadata is saved as human-readable JSON for easy inspection
        # without needing to load the full embedding artifacts.
        with open(

            self.metadata_file,

            "w"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

    def load(self):

        # Restore embeddings from disk — shape and dtype are preserved
        # exactly as saved, requiring no post-load transformation.
        embeddings = np.load(

            self.embedding_file

        )

        with open(

            self.id_file,

            "rb"

        ) as file:

            ids = pickle.load(file)

        # Deserialize the FAISS index with its full internal state intact,
        # ready for immediate nearest-neighbor search without retraining.
        index = faiss.read_index(

            str(self.faiss_file)

        )

        with open(

            self.metadata_file,

            "r"

        ) as file:

            metadata = json.load(file)

        # Return all four artifacts as a tuple so the caller can unpack
        # them in a single assignment matching the save() parameter order.
        return (

            embeddings,

            ids,

            index,

            metadata

        )