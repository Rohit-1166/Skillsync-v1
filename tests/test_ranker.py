import unittest
import numpy as np

from ranking.hybrid_ranker import HybridRanker
from models.candidate import Candidate, Profile, CareerEntry, Skill, Education, RecruiterSignals
from models.job_description import JobDescription

class MockEmbedder:
    def encode(self, text: str):
        return np.zeros(384, dtype=np.float32)

class MockFaissIndex:
    def __init__(self, ids: list[str]):
        self.ids = ids
    def search(self, embedding, top_k: int):
        # Return mock scores and indices for the two mock candidates
        scores = np.array([0.85, 0.75], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        return scores, indices

class TestRanker(unittest.TestCase):

    def setUp(self):
        # Create mock candidates corresponding to indices 0 and 1
        self.candidate1 = Candidate(
            candidate_id="CAND_001",
            profile=Profile(
                headline="Senior Engineer",
                summary="Experienced AI professional",
                location="Bengaluru",
                country="India",
                years_of_experience=7.0,
                current_title="Senior AI Engineer",
                current_company="Alpha Corp",
                current_industry="Information Technology"
            ),
            career_history=[],
            skills=[Skill(name="Python", proficiency="Expert", endorsements=5, duration_months=36)],
            education=[],
            signals=RecruiterSignals(profile_completeness_score=0.8, signup_date="", last_active_date="", open_to_work_flag=True, profile_views_received_30d=10, applications_submitted_30d=1, recruiter_response_rate=0.9, avg_response_time_hours=2.0, skill_assessment_scores={}, connection_count=100, endorsements_received=5, notice_period_days=30, expected_salary_min=15, expected_salary_max=25, preferred_work_mode="Hybrid", willing_to_relocate=True, github_activity_score=5.0, search_appearance_30d=50, saved_by_recruiters_30d=3, interview_completion_rate=1.0, offer_acceptance_rate=0.8, verified_email=True, verified_phone=True, linkedin_connected=True)
        )
        
        self.candidate2 = Candidate(
            candidate_id="CAND_002",
            profile=Profile(
                headline="Junior Engineer",
                summary="Entry level ML coder",
                location="Bengaluru",
                country="India",
                years_of_experience=1.5,
                current_title="Junior Engineer",
                current_company="Beta Corp",
                current_industry="Information Technology"
            ),
            career_history=[],
            skills=[Skill(name="Python", proficiency="Intermediate", endorsements=2, duration_months=12)],
            education=[],
            signals=RecruiterSignals(profile_completeness_score=0.7, signup_date="", last_active_date="", open_to_work_flag=True, profile_views_received_30d=2, applications_submitted_30d=1, recruiter_response_rate=0.5, avg_response_time_hours=48.0, skill_assessment_scores={}, connection_count=10, endorsements_received=1, notice_period_days=60, expected_salary_min=5, expected_salary_max=10, preferred_work_mode="Remote", willing_to_relocate=False, github_activity_score=1.0, search_appearance_30d=10, saved_by_recruiters_30d=0, interview_completion_rate=0.5, offer_acceptance_rate=0.5, verified_email=True, verified_phone=True, linkedin_connected=False)
        )
        
        self.candidates = [self.candidate1, self.candidate2]
        self.ids = ["CAND_001", "CAND_002"]
        
        self.mock_embedder = MockEmbedder()
        self.mock_faiss_index = MockFaissIndex(self.ids)
        
        self.ranker = HybridRanker(self.mock_embedder, self.mock_faiss_index)
        
        self.jd = JobDescription(
            raw_text="Lead AI Engineer with 5+ years experience",
            title="Lead AI Engineer",
            company="Redrob AI",
            location="Remote",
            work_mode="Hybrid",
            employment_type="Full-time",
            min_experience=5.0,
            max_experience=10.0,
            required_skills=["Python"],
            required_capabilities=["RAG"],
            hidden_expectations=[]
        )

    def test_rank_returns_sorted_candidates(self):
        ranked = self.ranker.rank(self.candidates, self.jd)
        
        self.assertEqual(len(ranked), 2)
        # CAND_001 has more experience and better signals, so it should rank higher
        self.assertEqual(ranked[0][0].candidate_id, "CAND_001")
        self.assertEqual(ranked[1][0].candidate_id, "CAND_002")
        
        # Check that scores are sorted in descending order
        self.assertGreater(ranked[0][1].final_score, ranked[1][1].final_score)
        
        # Verify similarity scores were properly populated
        self.assertAlmostEqual(ranked[0][1].similarity_score, 0.85)
        self.assertAlmostEqual(ranked[1][1].similarity_score, 0.75)

if __name__ == "__main__":
    unittest.main()
