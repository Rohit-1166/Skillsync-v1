import os

# Disables tokenizer multithreading to prevent excessive CPU contention
# when SentenceTransformers and PyTorch are processing large batches.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from pathlib import Path
import numpy as np  # pyright: ignore[reportMissingImports]

from config.settings import CANDIDATE_FILE, EMBEDDING_MODEL
from parser.candidate_parser import CandidateParser
from embeddings.semantic_document_builder import SemanticDocumentBuilder
from embeddings.embedder import Embedder
from retrieval.cache_manager import CacheManager
from retrieval.faiss_index import FaissIndex
from utils.logger import logger


class EmbeddingPipeline:

    def __init__(self):

        self.cache_manager = CacheManager()

        self.embedder = Embedder()

        self.candidate_file = CANDIDATE_FILE

    def run(self) -> tuple[FaissIndex, list]:

        logger.info("Initializing embedding pipeline...")

        parser = CandidateParser(self.candidate_file)

        candidates = parser.parse()

        # Reuse previously generated embeddings and FAISS index
        # to avoid rebuilding them on every application startup.
        if self._is_cache_valid():

            logger.info("Loading embeddings and FAISS index from cache...")

            try:

                _, ids, raw_index, _ = self.cache_manager.load()

                faiss_index = FaissIndex(self.embedder.dimension)

                faiss_index.index = raw_index

                faiss_index.ids = ids

                logger.info("Cache successfully loaded.")

                return faiss_index, candidates

            except Exception as e:

                logger.warning(f"Failed to load cache: {e}. Rebuilding index...")

        logger.info("Generating candidate embeddings (cache not found or empty)...")

        start_time = time.time()

        documents = []

        for candidate in candidates:

            doc = SemanticDocumentBuilder.build(candidate)

            documents.append(doc)

        # Large candidate datasets can exceed available RAM if encoded
        # in a single batch, so embeddings are generated in chunks.
        logger.info("Encoding documents in chunks of 10,000 to prevent RAM swapping...")

        chunk_size = 10000

        embeddings_list = []

        total_docs = len(documents)

        for i in range(0, total_docs, chunk_size):

            chunk = documents[i:i + chunk_size]

            chunk_idx = i // chunk_size + 1

            total_chunks = (total_docs + chunk_size - 1) // chunk_size

            logger.info(f"Encoding chunk {chunk_idx}/{total_chunks}...")

            chunk_start = time.time()

            chunk_embeddings = self.embedder.encode_batch(
                chunk,
                batch_size=128,
                max_length=128
            )

            embeddings_list.append(chunk_embeddings)

            chunk_elapsed = time.time() - chunk_start

            logger.info(f"Chunk {chunk_idx} completed in {chunk_elapsed:.2f} seconds.")

        logger.info("Aggregating embeddings...")

        # Merge all chunk embeddings into a single matrix
        # before building the FAISS index.
        embeddings = np.vstack(embeddings_list)

        logger.info("Building FAISS index...")

        faiss_index = FaissIndex(self.embedder.dimension)

        candidate_ids = [c.candidate_id for c in candidates]

        faiss_index.add_batch(candidate_ids, embeddings)

        logger.info("Saving generated embeddings and FAISS index to cache...")

        # Metadata helps verify cache compatibility between runs
        # and simplifies debugging when models or datasets change.
        metadata = {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": self.embedder.dimension,
            "candidate_count": len(candidates),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        }

        self.cache_manager.save(
            embeddings=embeddings,
            ids=candidate_ids,
            index=faiss_index.index,
            metadata=metadata
        )

        elapsed = time.time() - start_time

        logger.info(f"Pipeline generation completed in {elapsed:.2f} seconds.")

        return faiss_index, candidates

    def _is_cache_valid(self) -> bool:

        if not self.cache_manager.cache_exists():

            return False

        # Prevent loading partially written or corrupted cache files.
        cache_files = [
            self.cache_manager.embedding_file,
            self.cache_manager.id_file,
            self.cache_manager.faiss_file,
            self.cache_manager.metadata_file
        ]

        for file_path in cache_files:

            if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:

                return False

        return True