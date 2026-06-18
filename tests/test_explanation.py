import unittest

from reasoning.explanation_generator import ExplanationGenerator
from models.candidate import Candidate, Profile, CareerEntry, Skill, Education, RecruiterSignals
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures

class TestExplanation(unittest.TestCase):

    def setUp(self):
        # Create standard mock candidate
        self.candidate = Candidate(
            candidate_id="CAND_999",
            profile=Profile(
                headline="Lead Architect",
                summary="Expert in RAG and search engines",
                location="Delhi",
                country="India",
                years_of_experience=10.0,
                current_title="Lead Architect",
                current_company="Microsoft",
                current_industry="Information Technology"
            ),
            career_history=[
                CareerEntry(
                    company="Microsoft",
                    title="Lead Architect",
                    description="Engineered core semantic retrieval and embedding pipelines.",
                    duration_months=48,
                    industry="Information Technology",
                    company_size="10000+",
                    is_current=True
                )
            ],
            skills=[Skill(name="Python", proficiency="Expert", endorsements=12, duration_months=96)],
            education=[
                Education(
                    institution="IIT Madras",
                    degree="M.Tech",
                    field_of_study="Computer Science",
                    tier="tier_1",
                    grade="9.5/10"
                )
            ],
            signals=RecruiterSignals(
                profile_completeness_score=0.9,
                signup_date="",
                last_active_date="",
                open_to_work_flag=True,
                profile_views_received_30d=10,
                applications_submitted_30d=2,
                recruiter_response_rate=0.9,
                avg_response_time_hours=1.0,
                skill_assessment_scores={},
                connection_count=100,
                endorsements_received=5,
                notice_period_days=15,
                expected_salary_min=20,
                expected_salary_max=30,
                preferred_work_mode="Hybrid",
                willing_to_relocate=True,
                github_activity_score=5.0,
                search_appearance_30d=50,
                saved_by_recruiters_30d=5,
                interview_completion_rate=1.0,
                offer_acceptance_rate=0.9,
                verified_email=True,
                verified_phone=True,
                linkedin_connected=True
            )
        )

        # Setup standard mock job description
        self.jd = JobDescription(
            raw_text="Lead AI Engineer with 5-10 years experience",
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

        # Mock feature values
        self.features = CandidateFeatures()
        self.features.experience_score = 0.95
        self.features.experience_years_score = 0.90
        self.features.experience_fit_score = 1.00
        self.features.average_tenure_score = 0.95
        self.features.career_stability_score = 1.00
        
        self.features.skill_match_score = 0.85
        self.features.required_skills_match = 1.00
        self.features.preferred_skills_match = 0.70
        self.features.ai_technical_depth = 0.90
        self.features.backend_technical_depth = 0.80

        self.features.capability_match_score = 0.90
        self.features.required_capabilities_match = 1.00
        self.features.project_complexity_score = 0.85
        self.features.role_consistency_score = 0.85

        self.features.industry_score = 1.00
        
        self.features.education_score = 0.95
        self.features.education_tier_score = 1.00
        self.features.degree_relevance_score = 1.00
        self.features.education_grade_score = 0.85

        self.features.recruiter_signal_score = 0.90
        self.features.profile_completeness_score = 0.90
        self.features.recruiter_interest_score = 0.90
        self.features.candidate_responsiveness_score = 0.90

        self.features.notice_period_score = 0.95
        self.features.work_mode_alignment_score = 1.00
        self.features.career_growth_score = 0.80
        
        self.features.final_score = 0.92

    def test_report_generation_structure(self):
        report = ExplanationGenerator.generate(
            candidate=self.candidate,
            similarity=0.82,
            features=self.features,
            jd=self.jd
        )
        
        self.assertIsInstance(report, str)
        self.assertIn("## Candidate: CAND_999", report)
        self.assertIn("🟢 STRONG MATCH", report)
        self.assertIn("### Key Recruiter Signals", report)
        self.assertIn("### Score Breakdown", report)
        self.assertIn("### Extracted Evidence", report)
        
        # Verify specific details parsed correctly
        self.assertIn("Microsoft", report)
        self.assertIn("IIT Madras", report)
        self.assertIn("Tenure", report)

if __name__ == "__main__":
    unittest.main()
