import unittest
import numpy as np

from ranking.hybrid_ranker import HybridRanker
from models.candidate import Candidate, Profile, CareerEntry, Skill, Education, RecruiterSignals
from models.job_description import JobDescription


# Minimal embedder stub that returns a zero vector for any input.
# Decouples the ranker test from the sentence transformer model,
# allowing the hybrid scoring logic to be tested without GPU or
# model weight dependencies.
class MockEmbedder:
    def encode(self, text: str):
        return np.zeros(384, dtype=np.float32)


# Deterministic FAISS stub that returns fixed scores and indices
# regardless of the query embedding.
# Pins retrieval output so the test asserts ranker behavior,
# not FAISS approximate search variance.
class MockFaissIndex:
    def __init__(self, ids: list[str]):
        self.ids = ids

    def search(self, embedding, top_k: int):
        # Return mock scores and indices for the two mock candidates.
        # CAND_001 at index 0 receives a higher similarity score (0.85)
        # than CAND_002 at index 1 (0.75) to drive the expected sort order.
        scores = np.array([0.85, 0.75], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        return scores, indices


# Validates the HybridRanker's end-to-end behavior using controlled
# mock inputs, confirming that sorting, score propagation, and
# feature-weighted hybrid scoring all function correctly together.
class TestRanker(unittest.TestCase):

    def setUp(self):

        # CAND_001 is intentionally a stronger candidate: more experience,
        # better signals, and higher proficiency — ensuring the ranker
        # places it above CAND_002 when feature scores are combined
        # with the mocked similarity scores.
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
            # High engagement signals: strong response rate, short notice,
            # verified contact details, and active recruiter interest.
            signals=RecruiterSignals(
                profile_completeness_score=0.8, signup_date="", last_active_date="",
                open_to_work_flag=True, profile_views_received_30d=10,
                applications_submitted_30d=1, recruiter_response_rate=0.9,
                avg_response_time_hours=2.0, skill_assessment_scores={},
                connection_count=100, endorsements_received=5, notice_period_days=30,
                expected_salary_min=15, expected_salary_max=25,
                preferred_work_mode="Hybrid", willing_to_relocate=True,
                github_activity_score=5.0, search_appearance_30d=50,
                saved_by_recruiters_30d=3, interview_completion_rate=1.0,
                offer_acceptance_rate=0.8, verified_email=True,
                verified_phone=True, linkedin_connected=True
            )
        )

        # CAND_002 is a weaker candidate: low experience, poor signals,
        # and a long notice period — designed to rank below CAND_001
        # even though it also receives a non-zero mock similarity score.
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
            # Low engagement signals: slow response, long notice, no LinkedIn,
            # and minimal recruiter activity — expected to produce a lower feature score.
            signals=RecruiterSignals(
                profile_completeness_score=0.7, signup_date="", last_active_date="",
                open_to_work_flag=True, profile_views_received_30d=2,
                applications_submitted_30d=1, recruiter_response_rate=0.5,
                avg_response_time_hours=48.0, skill_assessment_scores={},
                connection_count=10, endorsements_received=1, notice_period_days=60,
                expected_salary_min=5, expected_salary_max=10,
                preferred_work_mode="Remote", willing_to_relocate=False,
                github_activity_score=1.0, search_appearance_30d=10,
                saved_by_recruiters_30d=0, interview_completion_rate=0.5,
                offer_acceptance_rate=0.5, verified_email=True,
                verified_phone=True, linkedin_connected=False
            )
        )

        self.candidates = [self.candidate1, self.candidate2]
        self.ids = ["CAND_001", "CAND_002"]

        self.mock_embedder = MockEmbedder()
        self.mock_faiss_index = MockFaissIndex(self.ids)

        # Inject mocks so the ranker exercises its scoring and sorting
        # logic without touching real embeddings or FAISS search.
        self.ranker = HybridRanker(self.mock_embedder, self.mock_faiss_index)

        # JD requires 5+ years experience and Python skills —
        # aligned with CAND_001 to reinforce the expected ranking order.
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

        # Assert both candidates survived retrieval and scoring
        # without being filtered by the honeypot check.
        self.assertEqual(len(ranked), 2)

        # CAND_001 has more experience and better signals so it should
        # rank higher even though both receive non-zero similarity scores.
        self.assertEqual(ranked[0][0].candidate_id, "CAND_001")
        self.assertEqual(ranked[1][0].candidate_id, "CAND_002")

        # Verify the list is sorted descending by hybrid score —
        # a violation here means the ranker's sort step is broken.
        self.assertGreater(ranked[0][1].final_score, ranked[1][1].final_score)

        # Confirm that mocked FAISS similarity scores are correctly
        # propagated into the features object after hybrid score computation.
        self.assertAlmostEqual(ranked[0][1].similarity_score, 0.85)
        self.assertAlmostEqual(ranked[1][1].similarity_score, 0.75)


if __name__ == "__main__":
    unittest.main()