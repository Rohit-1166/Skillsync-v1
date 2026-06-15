import json
import pickle
from pathlib import Path

import faiss # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]


class CacheManager:

    def __init__(self):

        self.cache_dir = Path("cache")

        self.cache_dir.mkdir(
            exist_ok=True
        )

        self.embedding_file = (
            self.cache_dir /
            "candidate_embeddings.npy"
        )

        self.id_file = (
            self.cache_dir /
            "candidate_ids.pkl"
        )

        self.faiss_file = (
            self.cache_dir /
            "faiss.index"
        )

        self.metadata_file = (
            self.cache_dir /
            "metadata.json"
        )

    def cache_exists(self):

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

        np.save(

            self.embedding_file,

            embeddings

        )

        with open(

            self.id_file,

            "wb"

        ) as file:

            pickle.dump(

                ids,

                file

            )

        faiss.write_index(

            index,

            str(self.faiss_file)

        )

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

        embeddings = np.load(

            self.embedding_file

        )

        with open(

            self.id_file,

            "rb"

        ) as file:

            ids = pickle.load(file)

        index = faiss.read_index(

            str(self.faiss_file)

        )

        with open(

            self.metadata_file,

            "r"

        ) as file:

            metadata = json.load(file)

        return (

            embeddings,

            ids,

            index,

            metadata

        )