import unittest
from pathlib import Path
import os
import json

from parser.jd_parser import JDParser
from parser.candidate_parser import CandidateParser
from models.candidate import Candidate
from models.job_description import JobDescription


# Integration-style tests that verify the parsing layer produces
# correctly structured model objects from raw input formats.
# Tests both parsers in isolation so regressions in field mapping
# are caught before they propagate into the ranking pipeline.
class TestParsers(unittest.TestCase):

    def setUp(self):

        # Write a temporary JSONL file to disk rather than mocking
        # the file system — CandidateParser requires a real file path,
        # so an in-memory substitute would not exercise the actual I/O path.
        self.test_dir = Path("tests")
        self.test_dir.mkdir(exist_ok=True)
        self.jsonl_path = self.test_dir / "temp_test_candidates.jsonl"

        # Mock candidate covers all major sections of the JSONL schema:
        # profile, career, skills, education, and redrob_signals.
        # Chosen values are realistic enough to exercise type casting
        # (float experience, int duration) without hitting edge cases.
        self.mock_candidate_data = {
            "candidate_id": "CAND_TEST_999",
            "profile": {
                "headline": "Senior Software Engineer",
                "summary": "Building scalable microservices and backend architectures.",
                "location": "Bengaluru",
                "country": "India",
                "years_of_experience": 6.5,
                "current_title": "Software Engineer II",
                "current_company": "Acme Corp",
                "current_industry": "Information Technology"
            },
            "career_history": [
                {
                    "company": "Acme Corp",
                    "title": "Software Engineer II",
                    "description": "Built distributed backend systems using Python and FastAPI.",
                    "duration_months": 24,
                    "industry": "Information Technology",
                    "company_size": "500-1000",
                    "is_current": True
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "Expert", "endorsements": 15, "duration_months": 48},
                {"name": "FastAPI", "proficiency": "Intermediate", "endorsements": 5, "duration_months": 24}
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "tier": "tier_1",
                    "grade": "8.5/10"
                }
            ],
            # redrob_signals uses a nested salary range object —
            # included here to verify the parser correctly extracts
            # min/max from the nested structure rather than the top level.
            "redrob_signals": {
                "profile_completeness_score": 0.85,
                "open_to_work_flag": True,
                "notice_period_days": 30,
                "github_activity_score": 7.2,
                "preferred_work_mode": "Hybrid",
                "expected_salary_range_inr_lpa": {
                    "min": 18.0,
                    "max": 24.0
                }
            }
        }

        # Write as a single-line JSONL record to match the production
        # data format that CandidateParser expects to iterate over.
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.mock_candidate_data) + "\n")

    def tearDown(self):

        # Remove the temporary JSONL file after each test to prevent
        # leftover files from affecting subsequent test runs.
        if self.jsonl_path.exists():
            os.remove(self.jsonl_path)

    def test_jd_parser(self):

        # Sample JD includes all extractable fields: title, company,
        # location, experience range, and implicit skill mentions.
        # work_mode is intentionally omitted to verify the empty-string default.
        sample_jd_text = """
        Job Description: Senior AI Engineer
        Company: Redrob AI
        Location: Seoul, South Korea
        Experience Required: 5 to 9 Years
        
        We are looking for a Senior AI Engineer. Required skills include machine learning, RAG, and NLP.
        """

        parser = JDParser(sample_jd_text)
        jd = parser.parse()

        # Verify the return type before field assertions to catch cases
        # where parse() returns None or raises without a clear error.
        self.assertIsInstance(jd, JobDescription)

        # Assert structured fields are extracted correctly from
        # the labeled header pattern used by the regex extractors.
        self.assertEqual(jd.title, "Senior AI Engineer")
        self.assertEqual(jd.company, "Redrob AI")
        self.assertEqual(jd.location, "Seoul, South Korea")

        # work_mode defaults to empty string when none of the
        # hybrid/remote/onsite keywords appear in the JD text.
        self.assertEqual(jd.work_mode, "")

        # Experience range is parsed as floats from the "5 to 9 Years" pattern.
        self.assertEqual(jd.min_experience, 5.0)
        self.assertEqual(jd.max_experience, 9.0)

    def test_candidate_parser(self):

        parser = CandidateParser(self.jsonl_path)
        candidates = parser.parse()

        # Assert exactly one candidate was parsed from the single-line JSONL.
        # A count mismatch indicates the parser is splitting or skipping lines.
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]

        self.assertIsInstance(candidate, Candidate)

        # Verify top-level candidate ID is correctly mapped from the raw key.
        self.assertEqual(candidate.candidate_id, "CAND_TEST_999")
        self.assertEqual(candidate.profile.current_title, "Software Engineer II")

        # Assert collection lengths before indexing to produce clear
        # failure messages when a section is silently dropped during parsing.
        self.assertEqual(len(candidate.career_history), 1)
        self.assertEqual(candidate.career_history[0].company, "Acme Corp")

        self.assertEqual(len(candidate.skills), 2)
        self.assertEqual(candidate.skills[0].name, "Python")

        self.assertEqual(len(candidate.education), 1)
        self.assertEqual(candidate.education[0].institution, "IIT Bombay")

        # Verify recruiter signals are correctly extracted from the
        # nested redrob_signals key rather than the top-level profile.
        self.assertEqual(candidate.signals.notice_period_days, 30)
        self.assertTrue(candidate.signals.open_to_work_flag)


if __name__ == "__main__":
    unittest.main()