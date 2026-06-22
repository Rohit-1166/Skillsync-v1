from models.candidate import Candidate
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures
from features.feature_engineering import FeatureEngineer


# Hybrid ranking engine. Runs FAISS first for recall, then feature scoring for precision.
class HybridRanker:

    def __init__(self, embedder, faiss_index):

        # Need to share embedder instance so vectors match
        self.embedder = embedder

        # Cached index for fast retrieval
        self.faiss_index = faiss_index

        # Load scoring rules once
        self.feature_engineer = FeatureEngineer()

    def rank(
        self,
        candidates: list[Candidate],
        jd: JobDescription,
        top_k_retrieval: int = 500,
        semantic_weight: float = 0.60
    ) -> list[tuple[Candidate, CandidateFeatures]]:
        """
        Ranks candidates using a hybrid approach:
        1. Retrieves the top 500 candidates using FAISS semantic search.
        2. Computes hand-engineered features for these top 500 candidates.
        3. Combines semantic similarity score and feature score using dynamic weights.

        Returns:
            list[tuple[Candidate, CandidateFeatures]]: Ranked candidates with their features.
        """

        # lookup map for fast FAISS id matching
        candidate_map = {c.candidate_id: c for c in candidates}

        # build single string for JD embedding
        jd_document = self._build_jd_document(jd)

        # embed JD as query vector
        jd_embedding = self.embedder.encode(jd_document)

        # Get more candidates than needed and rerank them later
        similarity_scores, indices = self.faiss_index.search(
            jd_embedding,
            top_k=top_k_retrieval
        )

        ranked = []

        for similarity, idx in zip(similarity_scores, indices):

            # Handle FAISS padding (-1)
            if idx == -1 or idx >= len(self.faiss_index.ids):
                continue

            candidate_id = self.faiss_index.ids[idx]

            candidate = candidate_map.get(candidate_id)

            if not candidate:
                continue

            # Filter honeypots
            is_honeypot, _ = self._is_honeypot(candidate)
            if is_honeypot:
                continue

            # Score features only on top FAISS results to save time
            features = self.feature_engineer.extract(
                candidate,
                jd
            )

            # Calculate final hybrid score
            hybrid_score = (
                semantic_weight * float(similarity)
                + (1.0 - semantic_weight) * features.final_score
            )

            # Override final score for UI/explanation downstream
            features.final_score = hybrid_score
            features.similarity_score = float(similarity)

            ranked.append((candidate, features))

        # Sort descending by final score
        ranked.sort(
            key=lambda x: x[1].final_score,
            reverse=True
        )

        return ranked

    def _build_jd_document(self, jd: JobDescription) -> str:
        """
        Creates a recruiter-style natural language description of the Job Description.
        """

        # Build the JD document to match candidate profile structure
        parts = []

        parts.append(jd.title)

        parts.extend(jd.required_skills)

        parts.extend(jd.required_capabilities)

        parts.extend(jd.hidden_expectations)

        return "\n".join(parts)

    def _is_honeypot(self, candidate: Candidate) -> tuple[bool, str]:
        """
        Detects subtly impossible profiles (honeypots) to filter them out of rankings.
        """
        from datetime import datetime

        # TODO: verify this check isn't too strict
        # Check 1: Expert proficiency with 0 months duration
        for s in candidate.skills:
            if s.duration_months == 0 and s.proficiency.lower() in ["expert", "advanced"]:
                return True, f"Skill anomaly: {s.name} is {s.proficiency} with 0 months"

        # Check 2: Single role duration > total experience
        years_exp = candidate.profile.years_of_experience
        for job in candidate.career_history:
            if job.duration_months / 12.0 > years_exp + 0.5:
                return True, f"Job duration anomaly: job {job.company} duration {job.duration_months / 12.0:.1f} yrs > total exp {years_exp:.1f} yrs"

        # Check 3: Validate actual dates against declared duration_months
        current_date = datetime.now()
        current_year = current_date.year
        oldest_year = current_year
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
                        end_dt = current_date
                    
                    actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)

                    # 6 month buffer for rounding errors
                    if duration > actual_months + 6: # 6 months buffer
                        return True, f"Date interval anomaly: job {job.company} duration_months {duration} > actual date interval {actual_months}"
                except Exception:
                    pass

        # Check 4: Check if total years experience exceeds possible years since oldest job
        if oldest_year < current_year:
            max_possible_exp = current_year - oldest_year
            if years_exp > max_possible_exp + 1.5:
                return True, f"Oldest job anomaly: years_of_experience {years_exp:.1f} > max possible {max_possible_exp:.1f} yrs (oldest job started in {oldest_year})"

        # Check 5: Education end year before start year
        for edu in candidate.education:
            start_y = edu.start_year
            end_y = edu.end_year
            if start_y and end_y and start_y > end_y:
                return True, f"Education anomaly: start_year {start_y} > end_year {end_y}"

        return False, ""