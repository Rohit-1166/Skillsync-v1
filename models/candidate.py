from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Profile:
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_industry: str

@dataclass
class CareerEntry:
    company: str
    title: str
    description: str
    duration_months: int
    industry: str
    company_size: str
    is_current: bool
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@dataclass
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int

@dataclass
class Education:
    institution: str
    degree: str
    field_of_study: str
    tier: str
    grade: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None

@dataclass
class RecruiterSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
@dataclass
class Candidate:
    candidate_id: str
    profile: Profile
    career_history: List[CareerEntry] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    signals: List[RecruiterSignals] = field(default_factory=list)