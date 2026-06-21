from models.candidate import Candidate
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures
from features.feature_engineering import FeatureEngineer


# Core ranking engine that combines FAISS semantic retrieval with
# hand-engineered feature scoring to produce a final hybrid ranking.
# Two-stage design improves both recall (FAISS) and precision (features).
class HybridRanker:

    def __init__(self, embedder, faiss_index):

        # Shared embedder instance reused across ranking and explanation
        # to ensure consistent vector representations throughout the pipeline.
        self.embedder = embedder

        # Pre-built FAISS index loaded from cache at startup.
        # Enables sub-second approximate nearest-neighbor retrieval
        # over large candidate pools without exhaustive comparison.
        self.faiss_index = faiss_index

        # Feature engineer is instantiated once and reused per rank call
        # to avoid re-loading scoring rules on every request.
        self.feature_engineer = FeatureEngineer()

    def rank(
        self,
        candidates: list[Candidate],
        jd: JobDescription,
        top_k_retrieval: int = 500
    ) -> list[tuple[Candidate, CandidateFeatures]]:
        """
        Ranks candidates using a hybrid approach:
        1. Retrieves the top 500 candidates using FAISS semantic search.
        2. Computes hand-engineered features for these top 500 candidates.
        3. Combines semantic similarity score (60%) and feature score (40%).

        Returns:
            list[tuple[Candidate, CandidateFeatures]]: Ranked candidates with their features.
        """

        # O(1) lookup table built once per rank call.
        # Avoids scanning the full candidate list for every FAISS result.
        candidate_map = {c.candidate_id: c for c in candidates}

        # Convert the structured JD into a natural language document
        # so it can be embedded in the same vector space as candidates.
        jd_document = self._build_jd_document(jd)

        # JD embedding serves as the query vector for FAISS retrieval.
        # All candidate similarity scores are computed relative to this vector.
        jd_embedding = self.embedder.encode(jd_document)

        # Retrieve a larger pool than the final top_k to improve recall.
        # Feature reranking within this pool produces better final results
        # than relying on semantic similarity alone.
        similarity_scores, indices = self.faiss_index.search(
            jd_embedding,
            top_k=top_k_retrieval
        )

        ranked = []

        for similarity, idx in zip(similarity_scores, indices):

            # FAISS returns -1 for padded slots when the index has
            # fewer candidates than top_k_retrieval requested.
            if idx == -1 or idx >= len(self.faiss_index.ids):
                continue

            candidate_id = self.faiss_index.ids[idx]

            candidate = candidate_map.get(candidate_id)

            if not candidate:
                continue

            # Filter synthetic or malformed profiles before scoring
            # to prevent honeypot candidates from polluting the ranking.
            if self._is_honeypot(candidate):
                continue

            # Feature engineering runs only on the FAISS-retrieved subset,
            # not the full candidate pool, keeping latency bounded.
            features = self.feature_engineer.extract(
                candidate,
                jd
            )

            # Hybrid score weights semantic alignment higher than features
            # because embedding similarity captures meaning that keyword
            # matching and structured rules cannot fully express.
            hybrid_score = (
                0.60 * float(similarity)
                + 0.40 * features.final_score
            )

            # Overwrite final_score with the hybrid value so downstream
            # consumers like the explanation generator use a unified score.
            features.final_score = hybrid_score
            features.similarity_score = float(similarity)

            ranked.append((candidate, features))

        # Final sort ensures the returned list is ordered by hybrid score
        # regardless of the order FAISS returned its results.
        ranked.sort(
            key=lambda x: x[1].final_score,
            reverse=True
        )

        return ranked

    def _build_jd_document(self, jd: JobDescription) -> str:
        """
        Creates a recruiter-style natural language description of the Job Description.
        """

        # Concatenates title, skills, capabilities, and culture signals
        # into a single document that mirrors how candidate profiles are built,
        # ensuring both are embedded in a comparable semantic space.
        parts = []

        parts.append(jd.title)

        parts.extend(jd.required_skills)

        parts.extend(jd.required_capabilities)

        parts.extend(jd.hidden_expectations)

        return "\n".join(parts)

    def _is_honeypot(self, candidate: Candidate) -> bool:
        """
        Detects subtly impossible profiles (honeypots) to filter them out of rankings.
        """
        from datetime import datetime

        # Check 1: Proficiency without duration is a data integrity red flag.
        # A candidate claiming expert-level skill with zero months of usage
        # indicates a synthetically generated or intentionally manipulated profile.
        for s in candidate.skills:
            if s.duration_months == 0 and s.proficiency.lower() in ["expert", "advanced"]:
                return True

        # Check 2: A single role lasting longer than total declared experience
        # is logically impossible and indicates fabricated career data.
        years_exp = candidate.profile.years_of_experience
        for job in candidate.career_history:
            if job.duration_months / 12.0 > years_exp + 0.5:
                return True

        # Check 3: Validate declared duration_months against actual start/end dates.
        # A 6-month buffer accommodates rounding and partial month differences.
        oldest_year = 2026
        for job in candidate.career_history:
            start_str = job.start_date
            end_str = job.end_date
            duration = job.duration_months
            
            if start_str:
                try:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    if start_dt.year < oldest_year:
                        oldest_year = start_dt.year
                    
                    if end_str:
                        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                    else:
                        # Treat open-ended roles as active through the current date.
                        end_dt = datetime(2026, 6, 16) # local time current date
                    
                    actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

                    # Flag profiles where declared duration significantly
                    # exceeds what the date range can physically support.
                    if duration > actual_months + 6: # 6 months buffer
                        return True
                except Exception:
                    pass

        # Check 4: If career started before 2026, verify declared experience
        # does not exceed the maximum years physically possible since then.
        if oldest_year < 2026:
            max_possible_exp = 2026 - oldest_year
            if years_exp > max_possible_exp + 1.5:
                return True

        # Check 5: Education with end year before start year is
        # chronologically invalid and indicates corrupted or fake data.
        for edu in candidate.education:
            start_y = edu.start_year
            end_y = edu.end_year
            if start_y and end_y and start_y > end_y:
                return True

        return False