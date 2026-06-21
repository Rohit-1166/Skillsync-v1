import faiss  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]


# Thin wrapper around a FAISS IndexFlatIP index that couples
# the vector index with a parallel ID list for candidate resolution.
# IndexFlatIP uses inner product similarity, which equals cosine similarity
# when embeddings are L2-normalized — as produced by the embedding pipeline.
class FaissIndex:

    def __init__(self, dimension):

        # IndexFlatIP performs exact brute-force inner product search.
        # Chosen over approximate methods (IVF, HNSW) to ensure no
        # valid candidate is missed during the retrieval stage.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Parallel list maintaining insertion-order mapping from
        # positional FAISS indices to candidate_id strings.
        # FAISS returns integer positions — this list resolves them to IDs.
        self.ids = []

    def add(
        self,
        candidate_id,
        embedding
    ):

        # FAISS requires a 2D float32 array even for single vectors.
        # Wrapping in a list and casting ensures the correct shape and dtype.
        vector = np.array(
            [embedding],
            dtype=np.float32
        )

        self.index.add(
            vector
        )

        # ID is appended after the vector so positional alignment
        # between self.ids and the FAISS index is always maintained.
        self.ids.append(
            candidate_id
        )

    def add_batch(
        self,
        candidate_ids,
        embeddings
    ):

        # Batch insertion is significantly faster than adding vectors
        # one at a time because FAISS can optimize memory allocation
        # and avoid repeated index resizing.
        vectors = np.array(
            embeddings,
            dtype=np.float32
        )

        self.index.add(
            vectors
        )

        self.ids.extend(
            candidate_ids
        )

    def search(
        self,
        embedding,
        top_k=100
    ):

        # FAISS search requires a 2D query matrix even for a single vector.
        # Shape is (1, dimension) — results are returned as (1, top_k) arrays.
        query = np.array(
            [embedding],
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query,
            top_k
        )

        # Unwrap the outer batch dimension since we always query one
        # JD at a time, returning flat arrays to the caller.
        return scores[0], indices[0]

    def save(
        self,
        path
    ):

        # Delegates serialization to FAISS's native binary format,
        # which preserves the full index state including trained parameters.
        faiss.write_index(
            self.index,
            str(path)
        )

    @staticmethod
    def load(path, dimension=384):

        # Reconstructs the wrapper object first so self.ids is initialized,
        # then overwrites the inner index with the deserialized FAISS index.
        # dimension default of 384 matches the BGE-small embedding size.
        index = FaissIndex(dimension)

        index.index = faiss.read_index(
            str(path)
        )

        return index