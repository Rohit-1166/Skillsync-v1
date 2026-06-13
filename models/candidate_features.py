from dataclasses import dataclass


@dataclass
class CandidateFeatures:
    # Baseline Features (preserved for compatibility)
    experience_score: float = 0.0
    skill_match_score: float = 0.0
    capability_match_score: float = 0.0
    industry_score: float = 0.0
    education_score: float = 0.0
    recruiter_signal_score: float = 0.0
    similarity_score: float = 0.0

    # 1. Experience & Stability
    experience_years_score: float = 0.0
    experience_fit_score: float = 0.0
    average_tenure_score: float = 0.0
    career_stability_score: float = 0.0

    # 2. Skill & Technical Depth
    required_skills_match: float = 0.0
    preferred_skills_match: float = 0.0
    ai_technical_depth: float = 0.0
    backend_technical_depth: float = 0.0

    # 3. Capability & Job Alignment
    required_capabilities_match: float = 0.0
    project_complexity_score: float = 0.0
    role_consistency_score: float = 0.0

    # 4. Education Quality & Tier
    education_tier_score: float = 0.0
    degree_relevance_score: float = 0.0
    education_grade_score: float = 0.0

    # 5. Recruiter Engagement
    profile_completeness_score: float = 0.0
    recruiter_interest_score: float = 0.0
    candidate_responsiveness_score: float = 0.0

    # 6. Logistics & Alignment
    notice_period_score: float = 0.0
    work_mode_alignment_score: float = 0.0
    career_growth_score: float = 0.0

    # Final Aggregated Feature Score
    final_score: float = 0.0