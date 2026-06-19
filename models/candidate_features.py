from dataclasses import dataclass


@dataclass
class CandidateFeatures:
    # Baseline Features (preserved for compatibility)
    # These aggregate scores are retained for compatibility
    # with existing ranking and scoring components.
    experience_score: float = 0.0
    skill_match_score: float = 0.0
    capability_match_score: float = 0.0
    industry_score: float = 0.0
    education_score: float = 0.0
    recruiter_signal_score: float = 0.0

    # Semantic similarity score generated during retrieval
    # and hybrid ranking.
    similarity_score: float = 0.0

    # 1. Experience & Stability
    experience_years_score: float = 0.0
    experience_fit_score: float = 0.0

    # Measures long-term employment consistency.
    average_tenure_score: float = 0.0
    career_stability_score: float = 0.0

    # 2. Skill & Technical Depth
    required_skills_match: float = 0.0
    preferred_skills_match: float = 0.0

    # Technical specialization indicators used to
    # evaluate depth beyond simple keyword matching.
    ai_technical_depth: float = 0.0
    backend_technical_depth: float = 0.0

    # 3. Capability & Job Alignment
    required_capabilities_match: float = 0.0

    # Measures evidence of complex projects and role relevance.
    project_complexity_score: float = 0.0
    role_consistency_score: float = 0.0

    # 4. Education Quality & Tier
    education_tier_score: float = 0.0
    degree_relevance_score: float = 0.0
    education_grade_score: float = 0.0

    # 5. Recruiter Engagement
    profile_completeness_score: float = 0.0

    # Signals derived from recruiter activity and
    # candidate communication behaviour.
    recruiter_interest_score: float = 0.0
    candidate_responsiveness_score: float = 0.0

    # 6. Logistics & Alignment
    # Hiring practicality factors such as availability
    # and work preference compatibility.
    notice_period_score: float = 0.0
    work_mode_alignment_score: float = 0.0
    career_growth_score: float = 0.0

    # Final Aggregated Feature Score
    # Composite score used by the ranking engine after
    # combining all feature categories.
    final_score: float = 0.0