import unittest

from features.feature_engineering import FeatureEngineer
from models.candidate import Candidate, Profile, CareerEntry, Skill, Education, RecruiterSignals
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures

class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.engineer = FeatureEngineer()
        
        # Setup standard mock candidate
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
                    tier="tier_1",
                    grade="9.2/10"
                )
            ],
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

        # Setup standard mock job description
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
        self.assertIsInstance(features, CandidateFeatures)
        
        # Test core score category boundaries
        self.assertTrue(0.0 <= features.experience_score <= 1.0)
        self.assertTrue(0.0 <= features.skill_match_score <= 1.0)
        self.assertTrue(0.0 <= features.capability_match_score <= 1.0)
        self.assertTrue(0.0 <= features.industry_score <= 1.0)
        self.assertTrue(0.0 <= features.education_score <= 1.0)
        self.assertTrue(0.0 <= features.recruiter_signal_score <= 1.0)
        self.assertTrue(0.0 <= features.final_score <= 1.0)

    def test_individual_features(self):
        features = self.engineer.extract(self.candidate, self.jd)
        
        # Verify subcomponents are mapped and non-none
        self.assertIsNotNone(features.experience_years_score)
        self.assertIsNotNone(features.experience_fit_score)
        self.assertIsNotNone(features.average_tenure_score)
        self.assertIsNotNone(features.career_stability_score)
        self.assertIsNotNone(features.required_skills_match)
        self.assertIsNotNone(features.preferred_skills_match)
        self.assertIsNotNone(features.ai_technical_depth)
        self.assertIsNotNone(features.backend_technical_depth)
        
        # Verify specific feature logic (e.g. Google company prestige, IIT Delhi university tier)
        self.assertGreater(features.experience_years_score, 0.0)
        self.assertEqual(features.education_tier_score, 1.0)  # IIT Delhi is Tier 1
        
if __name__ == "__main__":
    unittest.main()
