import csv


# Diagnostic output layer that dumps the full feature score breakdown
# for every ranked candidate into a CSV file.
# Used during development and evaluation to verify that individual
# sub-scores are behaving correctly before trusting the final hybrid ranking.
class DebugSubmissionWriter:

    @staticmethod
    def write(
        ranked_candidates,
        output_file,
        top_k=100
    ):
        """
        Writes the top candidate rankings along with their detailed feature scores
        (including all 20 advanced recruiter features and the 6 baseline scores)
        to a CSV file for verification and explanation.
        """
        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            # Header row mirrors the CandidateFeatures dataclass field order.
            # Grouped by scoring category so the CSV is human-navigable
            # without needing to cross-reference the feature engineering code.
            writer.writerow([
                "rank",
                "candidate_id",
                "final_score",
                "similarity_score",
                # Baseline category scores — weighted aggregates
                # of the detailed sub-scores listed below each group.
                "experience_score",
                "skill_match_score",
                "capability_match_score",
                "industry_score",
                "education_score",
                "recruiter_signal_score",
                # 1. Experience & Stability
                "experience_years_score",
                "experience_fit_score",
                "average_tenure_score",
                "career_stability_score",
                # 2. Skill & Technical Depth
                "required_skills_match",
                "preferred_skills_match",
                "ai_technical_depth",
                "backend_technical_depth",
                # 3. Capability & Alignment
                "required_capabilities_match",
                "project_complexity_score",
                "role_consistency_score",
                # 4. Education Quality
                "education_tier_score",
                "degree_relevance_score",
                "education_grade_score",
                # 5. Recruiter Engagement
                "profile_completeness_score",
                "recruiter_interest_score",
                "candidate_responsiveness_score",
                # 6. Logistics & Alignment
                "notice_period_score",
                "work_mode_alignment_score",
                "career_growth_score"
            ])

            # Slice to top_k before iterating to avoid writing
            # more rows than requested when the ranked list is larger.
            for rank, (
                candidate,
                features
            ) in enumerate(
                ranked_candidates[:top_k],
                start=1
            ):

                # Scores are rounded to 6 decimal places to balance
                # numerical precision with CSV file readability.
                # Sufficient for score comparison and ranking audits.
                writer.writerow([
                    rank,
                    candidate.candidate_id,
                    round(features.final_score, 6),
                    round(features.similarity_score, 6),
                    round(features.experience_score, 6),
                    round(features.skill_match_score, 6),
                    round(features.capability_match_score, 6),
                    round(features.industry_score, 6),
                    round(features.education_score, 6),
                    round(features.recruiter_signal_score, 6),
                    # 1. Experience & Stability
                    round(features.experience_years_score, 6),
                    round(features.experience_fit_score, 6),
                    round(features.average_tenure_score, 6),
                    round(features.career_stability_score, 6),
                    # 2. Skill & Technical Depth
                    round(features.required_skills_match, 6),
                    round(features.preferred_skills_match, 6),
                    round(features.ai_technical_depth, 6),
                    round(features.backend_technical_depth, 6),
                    # 3. Capability & Alignment
                    round(features.required_capabilities_match, 6),
                    round(features.project_complexity_score, 6),
                    round(features.role_consistency_score, 6),
                    # 4. Education Quality
                    round(features.education_tier_score, 6),
                    round(features.degree_relevance_score, 6),
                    round(features.education_grade_score, 6),
                    # 5. Recruiter Engagement
                    round(features.profile_completeness_score, 6),
                    round(features.recruiter_interest_score, 6),
                    round(features.candidate_responsiveness_score, 6),
                    # 6. Logistics & Alignment
                    round(features.notice_period_score, 6),
                    round(features.work_mode_alignment_score, 6),
                    round(features.career_growth_score, 6)
                ])