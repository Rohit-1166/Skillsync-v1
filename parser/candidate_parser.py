import json
from pathlib import Path
from typing import Iterator

from models.candidate import (
    Candidate,
    Profile,
    CareerEntry,
    Skill,
    Education,
    RecruiterSignals
)

from utils.logger import logger


# TODO: Move JSONL parsing to a separate data loader class later
class CandidateParser:

    def __init__(self, jsonl_path: Path):

        self.jsonl_path = Path(jsonl_path)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Candidate file not found: {self.jsonl_path}"
            )

        self.total_records = 0
        self.success_records = 0
        self.failed_records = 0

    def parse(self) -> list[Candidate]:

        logger.info("Parsing started...")

        candidates = []

        # print(f"DEBUG: Opening jsonl from {self.jsonl_path}")
        with self.jsonl_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, start=1):

                self.total_records += 1

                try:

                    raw = json.loads(line)

                    candidate = self._parse_candidate(raw)

                    candidates.append(candidate)

                    self.success_records += 1

                except Exception as e:

                    # print(f"DEBUG: Failed to parse line {line_number}: {e}")
                    self.failed_records += 1

                    logger.error(
                        f"Line {line_number}: {e}"
                    )

        logger.info(self.summary())

        return candidates

    def stream(self) -> Iterator[Candidate]:

        # FIXME: Memory usage gets high on large datasets, keep streaming mode for now
        with self.jsonl_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                try:

                    raw = json.loads(line)

                    yield self._parse_candidate(raw)

                except Exception:

                    # skip bad lines silently
                    continue

    def summary(self):

        return (
            f"\n"
            f"Total Records   : {self.total_records}\n"
            f"Parsed          : {self.success_records}\n"
            f"Failed          : {self.failed_records}\n"
        )

    def _parse_candidate(self, raw: dict) -> Candidate:

        return Candidate(

            candidate_id=raw.get("candidate_id", ""),

            profile=self._parse_profile(
                raw.get("profile", {})
            ),

            career_history=self._parse_career(
                raw.get("career_history", [])
            ),

            skills=self._parse_skills(
                raw.get("skills", [])
            ),

            education=self._parse_education(
                raw.get("education", [])
            ),

            signals=self._parse_signals(
                raw.get("redrob_signals", {})
            )
        )

    def _parse_profile(self, profile):

        return Profile(

            headline=profile.get("headline", ""),

            summary=profile.get("summary", ""),

            location=profile.get("location", ""),

            country=profile.get("country", ""),

            years_of_experience=float(
                profile.get(
                    "years_of_experience",
                    0
                )
            ),

            current_title=profile.get(
                "current_title",
                ""
            ),

            current_company=profile.get(
                "current_company",
                ""
            ),

            current_industry=profile.get(
                "current_industry",
                ""
            )
        )

    def _parse_career(self, history):

        careers = []

        for job in history:

            careers.append(

                CareerEntry(

                    company=job.get("company", ""),

                    title=job.get("title", ""),

                    description=job.get(
                        "description",
                        ""
                    ),

                    duration_months=int(
                        job.get(
                            "duration_months",
                            0
                        )
                    ),

                    industry=job.get(
                        "industry",
                        ""
                    ),

                    company_size=job.get(
                        "company_size",
                        ""
                    ),

                    is_current=job.get(
                        "is_current",
                        False
                    ),

                    start_date=job.get("start_date"),

                    end_date=job.get("end_date")

                )
            )

        return careers

    def _parse_skills(self, skills):

        parsed = []

        for skill in skills:

            parsed.append(

                Skill(

                    name=skill.get("name", ""),

                    proficiency=skill.get(
                        "proficiency",
                        ""
                    ),

                    endorsements=int(
                        skill.get(
                            "endorsements",
                            0
                        )
                    ),

                    duration_months=int(
                        skill.get(
                            "duration_months",
                            0
                        )
                    )
                )
            )

        return parsed

    def _parse_education(self, education):

        parsed = []

        for edu in education:

            start_y = edu.get("start_year")
            end_y = edu.get("end_year")

            parsed.append(

                Education(

                    institution=edu.get(
                        "institution",
                        ""
                    ),

                    degree=edu.get(
                        "degree",
                        ""
                    ),

                    field_of_study=edu.get(
                        "field_of_study",
                        ""
                    ),

                    tier=edu.get(
                        "tier",
                        ""
                    ),

                    grade=edu.get(
                        "grade",
                        ""
                    ),

                    start_year=int(start_y) if start_y is not None else None,

                    end_year=int(end_y) if end_y is not None else None

                )
            )

        return parsed

    def _parse_signals(self, signal):

        expected_salary = signal.get(
            "expected_salary_range_inr_lpa",
            {}
        )

        return RecruiterSignals(
            profile_completeness_score=float( signal.get("profile_completeness_score", 0) ),
            signup_date=signal.get( "signup_date", "" ),
            last_active_date=signal.get( "last_active_date", "" ),

            open_to_work_flag=bool( signal.get("open_to_work_flag", False) ),

            profile_views_received_30d=int( signal.get("profile_views_received_30d", 0) ),
            applications_submitted_30d=int( signal.get("applications_submitted_30d", 0) ),

            recruiter_response_rate=float( signal.get("recruiter_response_rate", 0) ),
            avg_response_time_hours=float( signal.get("avg_response_time_hours", 0) ),

            skill_assessment_scores=signal.get( "skill_assessment_scores", {} ),

            connection_count=int( signal.get("connection_count", 0) ),
            endorsements_received=int( signal.get("endorsements_received", 0) ),

            notice_period_days=int( signal.get("notice_period_days", 0) ),

            expected_salary_min=float( expected_salary.get("min", 0) ),
            expected_salary_max=float( expected_salary.get("max", 0) ),

            preferred_work_mode=signal.get( "preferred_work_mode", "" ),
            willing_to_relocate=bool( signal.get("willing_to_relocate", False) ),

            github_activity_score=float( signal.get("github_activity_score", -1) ),

            search_appearance_30d=int( signal.get("search_appearance_30d", 0) ),
            saved_by_recruiters_30d=int( signal.get("saved_by_recruiters_30d", 0) ),
            interview_completion_rate=float( signal.get("interview_completion_rate", 0) ),

            offer_acceptance_rate=float( signal.get("offer_acceptance_rate", -1) ),

            verified_email=bool( signal.get("verified_email", False) ),
            verified_phone=bool( signal.get("verified_phone", False) ),
            linkedin_connected=bool( signal.get("linkedin_connected", False) )
        )