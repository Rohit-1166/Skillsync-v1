import re
from models.candidate import Candidate
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures
from models.evidence import Evidence
from features.consistency import TimelineConsistencyAnalyzer
from knowledge.companies import get_company_tier_score
from knowledge.capabilities import CAPABILITY_MAP


# Generates two types of human-readable output from scoring results:
# 1. A concise CSV reasoning string for bulk submission exports.
# 2. A structured markdown report for individual candidate explanation.
# Acts as the final presentation layer in the SkillSync pipeline.
class ExplanationGenerator:

    @staticmethod
    def generate_csv_reasoning(
        candidate: Candidate,
        features: CandidateFeatures,
        jd: JobDescription
    ) -> str:
        """
        Generates a concise 1-2 sentence factual reasoning for the final submission CSV.
        """
        title = candidate.profile.current_title or "Engineer"
        exp = candidate.profile.years_of_experience

        highlights = []

        # Company pedigree is the strongest single signal for candidate quality.
        # Tier-1 company experience is surfaced first when present.
        from knowledge.companies import get_company_tier_score
        tier_1_cos = [job.company for job in candidate.career_history if get_company_tier_score(job.company) == 1.0]
        if tier_1_cos:
            highlights.append(f"ex-{tier_1_cos[0]}")
        elif features.education_tier_score == 1.0:
            # Fall back to education pedigree when no Tier-1 company history exists.
            highlights.append("Tier-1 education")

        # Technical depth flags indicate specialization beyond keyword matching.
        # AI depth is prioritized over backend depth as it is more role-specific.
        if features.ai_technical_depth >= 0.7:
            highlights.append("strong AI/NLP depth")
        elif features.backend_technical_depth >= 0.7:
            highlights.append("strong backend depth")

        # Extract recruiter engagement signals for the summary line.
        # Handles both list and direct object formats defensively.
        sig = candidate.signals
        response_rate = 0.0
        notice = 30
        if sig:
            sig_obj = sig[0] if isinstance(sig, list) and sig else sig
            response_rate = sig_obj.recruiter_response_rate
            notice = sig_obj.notice_period_days

        # Fallback highlight ensures the summary is never empty
        # even when no strong pedigree or depth signals are present.
        highlight_str = "; ".join(highlights) if highlights else "solid engineering fit"
        return f"{title} with {exp:.1f} yrs; {highlight_str}; response rate {response_rate:.2f} with {notice}d notice."


    @staticmethod
    def generate(
        candidate: Candidate,
        similarity: float,
        features: CandidateFeatures,
        jd: JobDescription
    ) -> str:
        """
        Generates a structured, highly polished recruiter narrative report for a candidate,
        detailing their overall score, match classification, category breakdowns,
        strengths/concerns, and extracted evidence.
        """
        # Recommendation level thresholds mirror the API response labels
        # so the markdown report and JSON output stay consistent.
        score = features.final_score
        if score >= 0.70:
            rec_level = "🟢 STRONG MATCH"
        elif score >= 0.60:
            rec_level = "🔵 GOOD MATCH"
        elif score >= 0.50:
            rec_level = "🟡 MARGINAL MATCH"
        else:
            rec_level = "🔴 UNALIGNED"

        lines = []
        lines.append(f"## Candidate: {candidate.candidate_id}")
        lines.append(f"- **Recommendation Level**: {rec_level}")
        lines.append(f"- **Final Hybrid Score**: `{score:.4f}`")
        lines.append(f"- **Semantic Context Similarity**: `{similarity:.4f}`")
        lines.append("")

        # Section 1: Strengths and concerns are derived from feature scores
        # rather than raw data so the narrative reflects the same signals
        # used during ranking — keeping scoring and explanation consistent.
        lines.append("### Key Recruiter Signals")
        strengths = []
        concerns = []

        # Experience evaluation checks both total volume and JD fit.
        # A candidate can have high experience but still miss the JD range.
        exp = candidate.profile.years_of_experience
        if features.experience_years_score >= 0.7:
            strengths.append(f"Extensive overall experience: **{exp:.1f} years** of work history.")
        elif exp < 2.0:
            concerns.append(f"Short overall experience: candidate has only **{exp:.1f} years** of experience.")

        if features.experience_fit_score >= 0.9:
            strengths.append(f"Perfect experience alignment: Candidate's {exp:.1f} years matches the JD range ({jd.min_experience}-{jd.max_experience} years).")
        elif exp < jd.min_experience:
            concerns.append(f"Experience gap: Candidate has {exp:.1f} years vs the required minimum of {jd.min_experience} years.")

        # Timeline analysis surfaces job hopping risk using actual
        # career dates rather than relying solely on the feature score.
        stats = TimelineConsistencyAnalyzer.analyze(candidate)
        avg_tenure = stats["average_tenure_months"]
        if features.career_stability_score >= 0.9:
            strengths.append(f"Excellent career stability: average tenure is **{avg_tenure / 12:.1f} years** per job with minimal hopping.")
        elif features.career_stability_score < 0.6:
            concerns.append(f"Job hopping risk: Candidate averages short tenures of **{avg_tenure:.1f} months** per job.")

        # Skill match percentage translates the raw overlap score
        # into a recruiter-friendly percentage for the report.
        if features.required_skills_match >= 0.7:
            strengths.append(f"Strong required skills alignment: matches **{features.required_skills_match * 100:.0f}%** of required technical keywords.")
        elif features.required_skills_match < 0.3:
            concerns.append(f"Technical keyword gaps: matches only **{features.required_skills_match * 100:.0f}%** of required skills.")

        # Technical depth scores go beyond keyword presence to reflect
        # demonstrated proficiency in AI/ML or backend systems.
        if features.ai_technical_depth >= 0.7:
            strengths.append("Demonstrates advanced AI technical depth: significant exposure to modern ML, RAG, and NLP architectures.")
        if features.backend_technical_depth >= 0.7:
            strengths.append("Strong systems and backend depth: experienced in microservices, databases, and deployment pipelines.")

        # Tier-1 company experience is a strong proxy for engineering quality
        # and is highlighted separately from skill or experience scores.
        tier_1_cos = [job.company for job in candidate.career_history if get_company_tier_score(job.company) == 1.0]
        if tier_1_cos:
            strengths.append(f"Prestige professional pedigree: worked at Tier 1 tech company ({', '.join(set(tier_1_cos[:2]))}).")

        # Education tier distinguishes elite institution graduates
        # and flags candidates with no academic credentials at all.
        if features.education_tier_score == 1.0:
            strengths.append("Elite academic credentials: Tier 1 educational institution graduate.")
        elif features.education_tier_score == 0.0:
            concerns.append("Academic profile missing: no educational credentials listed.")

        # Notice period and work mode are logistical signals that can
        # block a hire even when all technical signals are strong.
        if candidate.signals:
            sig = candidate.signals[0] if isinstance(candidate.signals, list) and candidate.signals else candidate.signals
            if sig.notice_period_days <= 15:
                strengths.append(f"High availability: short notice period of **{sig.notice_period_days} days**.")
            elif sig.notice_period_days >= 60:
                concerns.append(f"Joining delay: long notice period of **{sig.notice_period_days} days**.")

            if features.work_mode_alignment_score < 0.5:
                concerns.append(f"Work mode mismatch: Candidate prefers **{sig.preferred_work_mode}** but job is **{jd.work_mode}**.")

        if strengths:
            lines.append("#### 👍 Primary Strengths")
            for s in strengths:
                lines.append(f"- {s}")
        if concerns:
            lines.append("#### ⚠️ Potential Concerns")
            for c in concerns:
                lines.append(f"- {c}")
        lines.append("")

        # Section 2: Score breakdown table gives recruiters full visibility
        # into how each sub-score contributed to the final hybrid score.
        lines.append("### Score Breakdown")
        lines.append("| Category | Score | Detailed Components |")
        lines.append("| :--- | :---: | :--- |")
        lines.append(f"| **Experience & Stability** | `{features.experience_score:.2f}` | Years: {features.experience_years_score:.2f}, Fit: {features.experience_fit_score:.2f}, Tenure: {features.average_tenure_score:.2f}, Stability: {features.career_stability_score:.2f} |")
        lines.append(f"| **Skill & Technical Depth** | `{features.skill_match_score:.2f}` | Required: {features.required_skills_match:.2f}, Preferred: {features.preferred_skills_match:.2f}, AI Depth: {features.ai_technical_depth:.2f}, Backend Depth: {features.backend_technical_depth:.2f} |")
        lines.append(f"| **Capability & Job Alignment** | `{features.capability_match_score:.2f}` | Capabilities: {features.required_capabilities_match:.2f}, Scale Term Complexity: {features.project_complexity_score:.2f}, Role Consistency: {features.role_consistency_score:.2f} |")
        lines.append(f"| **Industry Alignment** | `{features.industry_score:.2f}` | Industry Match: {features.industry_score:.2f} |")
        lines.append(f"| **Education Quality** | `{features.education_score:.2f}` | Institution Tier: {features.education_tier_score:.2f}, Field Relevance: {features.degree_relevance_score:.2f}, Grades: {features.education_grade_score:.2f} |")
        lines.append(f"| **Recruiter Engagement** | `{features.recruiter_signal_score:.2f}` | Completeness: {features.profile_completeness_score:.2f}, Views/Saves Interest: {features.recruiter_interest_score:.2f}, Responsiveness: {features.candidate_responsiveness_score:.2f} |")
        lines.append(f"| **Logistics & Alignment** | `{features.notice_period_score:.2f}` | Notice Period: {features.notice_period_score:.2f}, Work Mode: {features.work_mode_alignment_score:.2f}, Title Progression Growth: {features.career_growth_score:.2f} |")
        lines.append("")

        # Section 3: Evidence table surfaces specific terms and companies
        # that justify capability claims, making the report auditable
        # and defensible to hiring managers reviewing the shortlist.
        lines.append("### Extracted Evidence")
        evidence_list = ExplanationGenerator._extract_evidence(candidate, jd)
        if not evidence_list:
            lines.append("*No significant brand or capability evidence found.*")
        else:
            lines.append("| Capability / Source | Evidence / Matched Terms | Confidence |")
            lines.append("| :--- | :--- | :---: |")
            for ev in evidence_list:
                # Cap displayed terms at 6 to keep the table readable
                # without hiding all matched evidence.
                terms_str = ", ".join(f"`{t}`" for t in ev.matched_terms[:6])
                lines.append(f"| **{ev.capability}** | {terms_str} <br> *({ev.source})* | `{ev.confidence:.2f}` |")
        lines.append("")
        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _extract_evidence(candidate: Candidate, jd: JobDescription) -> list[Evidence]:

        evidence_list = []

        # Combine profile summary and all career descriptions into one
        # searchable text block to maximize alias and evidence term coverage.
        combined_text = (candidate.profile.summary + " " + " ".join([j.description for j in candidate.career_history])).lower()

        # Section 1: Match candidate text against required JD capabilities only.
        # Skipping non-required capabilities keeps the evidence table focused
        # on what the JD actually demands rather than all possible skills.
        for cap_name, cap_data in CAPABILITY_MAP.items():
            is_required = any(req.lower() == cap_name.lower() for req in jd.required_capabilities)
            if not is_required:
                continue

            matched_terms = []
            for alias in cap_data["aliases"]:
                if alias.lower() in combined_text:
                    matched_terms.append(alias)
            for term in cap_data["evidence"]:
                if term.lower() in combined_text:
                    matched_terms.append(term)

            if matched_terms:
                matched_terms = list(set(matched_terms))

                # Confidence grows with the number of distinct matched terms,
                # capped at 1.0 to reflect maximum alignment with the capability.
                confidence = min(1.0, 0.5 + 0.1 * len(matched_terms))
                evidence_list.append(Evidence(
                    capability=cap_name.capitalize(),
                    confidence=confidence,
                    source="Career History & Summary Description",
                    matched_terms=matched_terms
                ))

        # Section 2: Company tier evidence provides pedigree context
        # separate from technical capability matching.
        for job in candidate.career_history:
            tier_score = get_company_tier_score(job.company)
            if tier_score == 1.0:
                evidence_list.append(Evidence(
                    capability="Tier-1 Company Experience",
                    confidence=1.0,
                    source=f"Role: {job.title} at {job.company}",
                    matched_terms=[job.company]
                ))
            elif tier_score == 0.7:
                evidence_list.append(Evidence(
                    capability="Tier-2 Company Experience",
                    confidence=0.8,
                    source=f"Role: {job.title} at {job.company}",
                    matched_terms=[job.company]
                ))

        # Section 3: Education pedigree evidence uses the precomputed
        # tier label rather than re-evaluating institution quality here.
        # Only the most recent or primary education record is checked.
        if candidate.education:
            edu = candidate.education[0]
            if "tier_1" in edu.tier.lower():
                evidence_list.append(Evidence(
                    capability="Elite Education Pedigree",
                    confidence=1.0,
                    source=f"{edu.degree} from {edu.institution}",
                    matched_terms=[edu.institution, edu.tier]
                ))
            elif "tier_2" in edu.tier.lower():
                evidence_list.append(Evidence(
                    capability="Strong Education Pedigree",
                    confidence=0.8,
                    source=f"{edu.degree} from {edu.institution}",
                    matched_terms=[edu.institution, edu.tier]
                ))

        return evidence_list