from models.candidate import Candidate
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures
from features.feature_engineering import FeatureEngineer


class HybridRanker:

    def __init__(self, embedder, faiss_index):

        self.embedder = embedder

        self.faiss_index = faiss_index

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

        # Map candidate_id to Candidate object for fast robust lookups
        candidate_map = {c.candidate_id: c for c in candidates}

        # Convert Job Description to a semantic document for embedding
        jd_document = self._build_jd_document(jd)

        # Generate JD embedding
        jd_embedding = self.embedder.encode(jd_document)

        # Retrieve top candidate indices and similarity scores
        similarity_scores, indices = self.faiss_index.search(
            jd_embedding,
            top_k=top_k_retrieval
        )

        ranked = []

        for similarity, idx in zip(similarity_scores, indices):

            if idx == -1 or idx >= len(self.faiss_index.ids):
                continue

            candidate_id = self.faiss_index.ids[idx]

            candidate = candidate_map.get(candidate_id)

            if not candidate:
                continue

            if self._is_honeypot(candidate):
                continue

            # Run feature engineering only on retrieved candidates
            features = self.feature_engineer.extract(
                candidate,
                jd
            )

            # Calculate the hybrid score: 60% semantic + 40% engineered features
            hybrid_score = (
                0.60 * float(similarity)
                + 0.40 * features.final_score
            )

            # Store the hybrid score in final_score to integrate with downstream writers
            features.final_score = hybrid_score
            features.similarity_score = float(similarity)

            ranked.append((candidate, features))

        # Sort candidates by the hybrid score in descending order
        ranked.sort(
            key=lambda x: x[1].final_score,
            reverse=True
        )

        return ranked

    def _build_jd_document(self, jd: JobDescription) -> str:
        """
        Creates a recruiter-style natural language description of the Job Description.
        """

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

        # 1. Skill duration anomaly (expert/advanced with 0 duration)
        for s in candidate.skills:
            if s.duration_months == 0 and s.proficiency.lower() in ["expert", "advanced"]:
                return True

        # 2. Single job duration > total experience
        years_exp = candidate.profile.years_of_experience
        for job in candidate.career_history:
            if job.duration_months / 12.0 > years_exp + 0.5:
                return True

        # 3. Date range anomaly in job history
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
                        end_dt = datetime(2026, 6, 16) # local time current date
                    
                    actual_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    if duration > actual_months + 6: # 6 months buffer
                        return True
                except Exception:
                    pass
                    
        if oldest_year < 2026:
            max_possible_exp = 2026 - oldest_year
            if years_exp > max_possible_exp + 1.5:
                return True

        # 4. Education year anomaly
        for edu in candidate.education:
            start_y = edu.start_year
            end_y = edu.end_year
            if start_y and end_y and start_y > end_y:
                return True

        return False