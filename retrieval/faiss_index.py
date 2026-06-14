import faiss  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]


class FaissIndex:

    def __init__(self, dimension):

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.ids = []

    def add(
        self,
        candidate_id,
        embedding
    ):

        vector = np.array(
            [embedding],
            dtype=np.float32
        )

        self.index.add(
            vector
        )

        self.ids.append(
            candidate_id
        )

    def add_batch(
        self,
        candidate_ids,
        embeddings
    ):

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

        query = np.array(
            [embedding],
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query,
            top_k
        )

        return scores[0], indices[0]

    def save(
        self,
        path
    ):

        faiss.write_index(
            self.index,
            str(path)
        )

    @staticmethod
    def load(path, dimension=384):

        index = FaissIndex(dimension)

        index.index = faiss.read_index(
            str(path)
        )

        return index