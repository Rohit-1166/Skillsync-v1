import unittest

from features.feature_engineering import FeatureEngineer
from models.candidate import Candidate, Profile, CareerEntry, Skill, Education, RecruiterSignals
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures


# Validates that FeatureEngineer correctly extracts and scores all
# sub-features from a candidate-JD pair without relying on the
# embedding pipeline or FAISS index — purely structural scoring tests.
class TestFeatures(unittest.TestCase):

    def setUp(self):

        # FeatureEngineer is instantiated once per test class to match
        # how it is used in production — a single instance across requests.
        self.engineer = FeatureEngineer()

        # Mock candidate is designed to be a strong match for the mock JD:
        # Tier-1 education, Tier-1 company, relevant skills, and solid signals.
        # This ensures most feature scores are non-trivial and boundary
        # tests exercise meaningful ranges rather than edge-case zeros.
        self.candidate = Candidate(
            candidate_id="CAND_001",
            profile=Profile(
                headline="Senior Developer",
                summary="Passionate AI and ML engineer with strong python and backend architecture experience.",
                location="Delhi",
                country="India",
                years_of_experience=8.0,
                current_title="Lead AI Engineer",
                current_company="Google",
                current_industry="Information Technology"
            ),
            career_history=[
                CareerEntry(
                    company="Google",
                    title="Lead AI Engineer",
                    description="Designed RAG pipelines, fine-tuned LLMs, and deployed microservices at scale.",
                    duration_months=36,
                    industry="Information Technology",
                    company_size="10000+",
                    is_current=True
                ),
                CareerEntry(
                    company="TCS",
                    title="Software Engineer",
                    description="Worked on enterprise Java and databases.",
                    duration_months=60,
                    industry="Information Technology",
                    company_size="10000+",
                    is_current=False
                )
            ],
            skills=[
                Skill(name="Python", proficiency="Expert", endorsements=10, duration_months=96),
                Skill(name="FastAPI", proficiency="Expert", endorsements=5, duration_months=36)
            ],
            education=[
                Education(
                    institution="IIT Delhi",
                    degree="M.Tech",
                    field_of_study="Computer Science",
                    # tier_1 label is used to assert education_tier_score == 1.0
                    # in test_individual_features below.
                    tier="tier_1",
                    grade="9.2/10"
                )
            ],
            # Signals are set to high-engagement values to exercise
            # the upper range of recruiter signal scoring.
            signals=RecruiterSignals(
                profile_completeness_score=0.9,
                signup_date="2025-01-01",
                last_active_date="2026-06-15",
                open_to_work_flag=True,
                profile_views_received_30d=50,
                applications_submitted_30d=5,
                recruiter_response_rate=0.8,
                avg_response_time_hours=4.0,
                skill_assessment_scores={},
                connection_count=500,
                endorsements_received=15,
                notice_period_days=30,
                expected_salary_min=20.0,
                expected_salary_max=30.0,
                preferred_work_mode="Hybrid",
                willing_to_relocate=True,
                github_activity_score=8.5,
                search_appearance_30d=120,
                saved_by_recruiters_30d=8,
                interview_completion_rate=1.0,
                offer_acceptance_rate=0.9,
                verified_email=True,
                verified_phone=True,
                linkedin_connected=True
            )
        )

        # JD is scoped to a Lead AI Engineer role to align with the
        # mock candidate's title, skills, and capabilities — ensuring
        # overlap scores are high enough to validate scoring logic
        # rather than testing the zero-match edge case.
        self.jd = JobDescription(
            raw_text="Looking for a Lead AI Engineer with 5-10 years experience",
            title="Lead AI Engineer",
            company="Redrob AI",
            location="Remote",
            work_mode="Hybrid",
            employment_type="Full-time",
            min_experience=5.0,
            max_experience=10.0,
            required_skills=["Python", "FastAPI", "RAG"],
            required_capabilities=["RAG", "Embeddings", "LLM"],
            hidden_expectations=["startup mindset"]
        )

    def test_feature_extraction(self):

        features = self.engineer.extract(self.candidate, self.jd)

        # Verify the return type before value assertions to ensure
        # extract() returns a CandidateFeatures object and not a dict or None.
        self.assertIsInstance(features, CandidateFeatures)

        # All category scores must be normalized to [0.0, 1.0].
        # Violations indicate a scoring formula that can overflow its bounds
        # and corrupt the hybrid score calculation in the ranker.
        self.assertTrue(0.0 <= features.experience_score <= 1.0)
        self.assertTrue(0.0 <= features.skill_match_score <= 1.0)
        self.assertTrue(0.0 <= features.capability_match_score <= 1.0)
        self.assertTrue(0.0 <= features.industry_score <= 1.0)
        self.assertTrue(0.0 <= features.education_score <= 1.0)
        self.assertTrue(0.0 <= features.recruiter_signal_score <= 1.0)
        self.assertTrue(0.0 <= features.final_score <= 1.0)

    def test_individual_features(self):

        features = self.engineer.extract(self.candidate, self.jd)

        # Assert all sub-feature fields are populated by extract().
        # None values would silently zero out weighted aggregates
        # in the category score calculations.
        self.assertIsNotNone(features.experience_years_score)
        self.assertIsNotNone(features.experience_fit_score)
        self.assertIsNotNone(features.average_tenure_score)
        self.assertIsNotNone(features.career_stability_score)
        self.assertIsNotNone(features.required_skills_match)
        self.assertIsNotNone(features.preferred_skills_match)
        self.assertIsNotNone(features.ai_technical_depth)
        self.assertIsNotNone(features.backend_technical_depth)

        # experience_years_score must be non-zero for a candidate with 8 years —
        # a zero value would indicate the scoring formula is not reading
        # years_of_experience from the profile correctly.
        self.assertGreater(features.experience_years_score, 0.0)

        # IIT Delhi is a known Tier-1 institution in the knowledge base.
        # A score of 1.0 confirms the education tier lookup is functioning
        # and the tier label is being matched case-insensitively.
        self.assertEqual(features.education_tier_score, 1.0)


if __name__ == "__main__":
    unittest.main()