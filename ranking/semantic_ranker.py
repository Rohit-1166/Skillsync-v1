import numpy as np # pyright: ignore[reportMissingImports]

from embeddings.semantic_document_builder import SemanticDocumentBuilder
from features.feature_engineering import FeatureEngineer


# Lightweight semantic ranker used independently of the HybridRanker.
# Shares the same two-stage retrieval strategy but operates directly
# on a preloaded candidate_map rather than a full candidate list,
# making it suitable for lower-latency or single-session ranking flows.
class SemanticRanker:

    def __init__(
        self,
        embedder,
        faiss_index,
        candidate_map
    ):
        # Shared embedder encodes both JD and candidate documents
        # into the same vector space for meaningful similarity comparison.
        self.embedder = embedder

        # FAISS index loaded from cache — avoids recomputing
        # candidate embeddings on every ranking request.
        self.faiss_index = faiss_index

        # Direct index-to-candidate map allows O(1) lookup
        # from FAISS result indices without scanning the full pool.
        self.candidate_map = candidate_map

        # Feature engineer is instantiated once and reused
        # to avoid reinitializing scoring rules per request.
        self.feature_engineer = FeatureEngineer()

    def rank(
        self,
        jd,
        top_k=100
    ):

        # Build a unified natural language document from the JD
        # so it can be embedded in the same space as candidate profiles.
        jd_document = self._build_jd_document(jd)

        # JD embedding acts as the FAISS query vector.
        jd_embedding = self.embedder.encode(jd_document)

        # Retrieve a broader pool than top_k to improve final ranking quality.
        # Reranking 500 candidates with feature scores produces better
        # top_k results than returning the raw FAISS top_k directly.
        similarity_scores, indices = self.faiss_index.search(
            jd_embedding,
            top_k=500
        )

        results = []

        for similarity, index in zip(similarity_scores, indices):

            # FAISS returns -1 for unfilled slots when the index
            # contains fewer entries than the requested top_k.
            if index == -1:
                continue

            candidate = self.candidate_map[index]

            # Feature extraction runs only on the FAISS-retrieved subset
            # to keep scoring cost proportional to retrieval pool size.
            features = self.feature_engineer.extract(
                candidate,
                jd
            )

            # Hybrid score mirrors the HybridRanker weighting:
            # semantic alignment is weighted higher as it captures
            # meaning that structured feature rules cannot fully express.
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

        # Sort descending by hybrid score before slicing to top_k,
        # ensuring the best candidates surface regardless of FAISS order.
        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:top_k]

    def _build_jd_document(
        self,
        jd
    ):

        # Concatenates structured JD fields into a single recruiter-style
        # document that mirrors the format used to build candidate embeddings,
        # ensuring both sides of the similarity comparison are comparable.
        parts = []

        parts.append(jd.title)

        parts.extend(jd.required_skills)

        parts.extend(jd.required_capabilities)

        parts.extend(jd.hidden_expectations)

        return "\n".join(parts)