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


# Entry point for loading structured candidate data from JSONL format.
# Each line in the file represents one candidate record as a JSON object.
# Keeps parsing logic isolated from the rest of the pipeline.
class CandidateParser:

    def __init__(self, jsonl_path: Path):

        self.jsonl_path = Path(jsonl_path)

        # Fail early if the data file is missing rather than
        # silently returning an empty candidate pool during ranking.
        if not self.jsonl_path.exists():
            raise FileNotFoundError(
                f"Candidate file not found: {self.jsonl_path}"
            )

        # Counters tracked across parse() to surface data quality issues
        # in the summary without interrupting the parsing loop.
        self.total_records = 0
        self.success_records = 0
        self.failed_records = 0

    def parse(self) -> list[Candidate]:

        logger.info("Parsing started...")

        candidates = []

        # Open with explicit UTF-8 encoding to handle
        # multilingual names, institutions, and company names.
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

                    # Log and skip malformed lines so a single bad record
                    # does not abort the entire candidate pool load.
                    self.failed_records += 1

                    logger.error(
                        f"Line {line_number}: {e}"
                    )

        logger.info(self.summary())

        return candidates

    def stream(self) -> Iterator[Candidate]:

        # Streaming mode keeps memory usage bounded when processing
        # large candidate datasets that cannot fit in memory at once.
        # Used by the embedding pipeline during index construction.
        with self.jsonl_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                try:

                    raw = json.loads(line)

                    yield self._parse_candidate(raw)

                except Exception:

                    # Silently skip corrupt lines in streaming mode
                    # to avoid breaking generator consumers mid-iteration.
                    continue

    def summary(self):

        # Structured parse report surfaced after batch loading.
        # Helps diagnose upstream data quality issues in the JSONL export.
        return (
            f"\n"
            f"Total Records   : {self.total_records}\n"
            f"Parsed          : {self.success_records}\n"
            f"Failed          : {self.failed_records}\n"
        )

    def _parse_candidate(self, raw: dict) -> Candidate:

        # Delegates each section of the raw JSON to a dedicated parser.
        # Keeps field-level parsing logic modular and independently testable.
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

            # Recruiter signals are stored under a platform-specific key
            # and mapped to a normalized internal model for ranking.
            signals=self._parse_signals(
                raw.get("redrob_signals", {})
            )
        )

    def _parse_profile(self, profile):

        # Core identity and career snapshot used by both the ranker
        # and the explanation generator to describe the candidate.
        return Profile(

            headline=profile.get("headline", ""),

            summary=profile.get("summary", ""),

            location=profile.get("location", ""),

            country=profile.get("country", ""),

            # Cast to float to support fractional experience values
            # such as 2.5 years returned by some data sources.
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

        # Career history is the primary signal for domain relevance
        # and seniority scoring in the feature engineering engine.
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

                    # Duration in months allows experience to be
                    # aggregated and compared across roles numerically.
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

                    # Company size informs seniority context scoring —
                    # startup vs enterprise experience carries different weight.
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

        # Skills feed directly into keyword overlap and proficiency
        # scoring during feature engineering against the JD.
        parsed = []

        for skill in skills:

            parsed.append(

                Skill(

                    name=skill.get("name", ""),

                    proficiency=skill.get(
                        "proficiency",
                        ""
                    ),

                    # Endorsement count acts as a social proof signal
                    # for skill credibility beyond self-reporting.
                    endorsements=int(
                        skill.get(
                            "endorsements",
                            0
                        )
                    ),

                    # Duration anchors skill depth — used to distinguish
                    # actively used skills from briefly mentioned ones.
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

            # Extract years separately to handle null values before
            # casting — avoids int(None) TypeErrors on incomplete records.
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

                    # Institution tier is a precomputed prestige label
                    # used by the feature engineer to score education quality.
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

        # Salary range is nested under its own key in the raw data.
        # Extracted once here to avoid repeated .get() chains below.
        expected_salary = signal.get(
            "expected_salary_range_inr_lpa",
            {}
        )

        # RecruiterSignals bundles behavioral, engagement, and
        # availability metadata used alongside semantic scores
        # to produce a more complete hiring signal for the ranker.
        return RecruiterSignals(
            profile_completeness_score=float( signal.get("profile_completeness_score", 0) ),
            signup_date=signal.get( "signup_date", "" ),
            last_active_date=signal.get( "last_active_date", "" ),

            # open_to_work_flag enables pre-filtering candidates
            # who are actively seeking roles before ranking begins.
            open_to_work_flag=bool( signal.get("open_to_work_flag", False) ),

            profile_views_received_30d=int( signal.get("profile_views_received_30d", 0) ),
            applications_submitted_30d=int( signal.get("applications_submitted_30d", 0) ),

            # Response rate and time inform recruiter reachability —
            # high scores indicate candidates likely to engage quickly.
            recruiter_response_rate=float( signal.get("recruiter_response_rate", 0) ),
            avg_response_time_hours=float( signal.get("avg_response_time_hours", 0) ),

            # Assessment scores are stored as a dict keyed by skill name,
            # allowing flexible lookup during feature engineering.
            skill_assessment_scores=signal.get( "skill_assessment_scores", {} ),

            connection_count=int( signal.get("connection_count", 0) ),
            endorsements_received=int( signal.get("endorsements_received", 0) ),

            # Notice period directly affects hiring timelines
            # and is surfaced in the explanation report.
            notice_period_days=int( signal.get("notice_period_days", 0) ),

            # Min/max salary extracted from the nested range object
            # for direct comparison against JD budget constraints.
            expected_salary_min=float( expected_salary.get("min", 0) ),
            expected_salary_max=float( expected_salary.get("max", 0) ),

            preferred_work_mode=signal.get( "preferred_work_mode", "" ),
            willing_to_relocate=bool( signal.get("willing_to_relocate", False) ),

            # -1 sentinel indicates no GitHub data is available,
            # distinguishing missing data from a zero-activity score.
            github_activity_score=float( signal.get("github_activity_score", -1) ),

            search_appearance_30d=int( signal.get("search_appearance_30d", 0) ),
            saved_by_recruiters_30d=int( signal.get("saved_by_recruiters_30d", 0) ),
            interview_completion_rate=float( signal.get("interview_completion_rate", 0) ),

            # -1 sentinel used here as well to flag candidates
            # with no offer history vs those with a 0% acceptance rate.
            offer_acceptance_rate=float( signal.get("offer_acceptance_rate", -1) ),

            # Verification flags contribute to profile trust scoring
            # and are displayed in the candidate explanation report.
            verified_email=bool( signal.get("verified_email", False) ),
            verified_phone=bool( signal.get("verified_phone", False) ),
            linkedin_connected=bool( signal.get("linkedin_connected", False) )
        )