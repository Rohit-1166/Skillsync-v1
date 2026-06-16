import numpy as np # pyright: ignore[reportMissingImports]

from embeddings.semantic_document_builder import SemanticDocumentBuilder
from features.feature_engineering import FeatureEngineer


class SemanticRanker:

    def __init__(
        self,
        embedder,
        faiss_index,
        candidate_map
    ):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.candidate_map = candidate_map
        self.feature_engineer = FeatureEngineer()

    def rank(
        self,
        jd,
        top_k=100
    ):

        # Convert JD into semantic text
        jd_document = self._build_jd_document(jd)

        # Embed JD
        jd_embedding = self.embedder.encode(jd_document)

        # Retrieve top candidates
        similarity_scores, indices = self.faiss_index.search(
            jd_embedding,
            top_k=500
        )

        results = []

        for similarity, index in zip(similarity_scores, indices):

            if index == -1:
                continue

            candidate = self.candidate_map[index]

            features = self.feature_engineer.extract(
                candidate,
                jd
            )

            final_score = (
                0.60 * float(similarity)
                + 0.40 * features.final_score
            )

            results.append(
                (
                    candidate,
                    final_score,
                    similarity,
                    features
                )
            )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]

    def _build_jd_document(
        self,
        jd
    ):

        parts = []

        parts.append(jd.title)

        parts.extend(jd.required_skills)

        parts.extend(jd.required_capabilities)

        parts.extend(jd.hidden_expectations)

        return "\n".join(parts)