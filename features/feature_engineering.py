import re
from models.candidate import Candidate
from models.job_description import JobDescription
from models.candidate_features import CandidateFeatures
from features.consistency import TimelineConsistencyAnalyzer
from knowledge.companies import get_company_tier_score
from knowledge.industries import get_industry_relevance_score


class FeatureEngineer:

    def extract(
        self,
        candidate: Candidate,
        jd: JobDescription
    ) -> CandidateFeatures:
        """
        Extracts 20 recruiter-inspired features and aggregates them into the 
        6 baseline categories and a final hybrid score.
        """
        features = CandidateFeatures()

        # 1. Experience & Stability
        features.experience_years_score = self._experience_years_score(candidate)
        features.experience_fit_score = self._experience_fit_score(candidate, jd)
        features.average_tenure_score = self._average_tenure_score(candidate)
        features.career_stability_score = self._career_stability_score(candidate)

        # 2. Skill & Technical Depth
        features.required_skills_match = self._required_skills_match(candidate, jd)
        features.preferred_skills_match = self._preferred_skills_match(candidate, jd)
        features.ai_technical_depth = self._ai_technical_depth(candidate)
        features.backend_technical_depth = self._backend_technical_depth(candidate)

        # 3. Capability & Job Alignment
        features.required_capabilities_match = self._required_capabilities_match(candidate, jd)
        features.project_complexity_score = self._project_complexity_score(candidate)
        features.role_consistency_score = self._role_consistency_score(candidate, jd)

        # 4. Education Quality & Tier
        features.education_tier_score = self._education_tier_score(candidate)
        features.degree_relevance_score = self._degree_relevance_score(candidate)
        features.education_grade_score = self._education_grade_score(candidate)

        # 5. Recruiter Engagement
        features.profile_completeness_score = self._profile_completeness_score(candidate)
        features.recruiter_interest_score = self._recruiter_interest_score(candidate)
        features.candidate_responsiveness_score = self._candidate_responsiveness_score(candidate)

        # 6. Logistics & Alignment
        features.notice_period_score = self._notice_period_score(candidate)
        features.work_mode_alignment_score = self._work_mode_alignment_score(candidate, jd)
        features.career_growth_score = self._career_growth_score(candidate)

        # =====================================================================
        # Aggregate categories into baseline features for backward compatibility
        # =====================================================================

        # Balance overall experience with how closely the candidate
        # matches the experience range specified in the JD.
        # Experience Score (20% weight in final features)
        features.experience_score = (
            0.50 * features.experience_years_score +
            0.50 * features.experience_fit_score
        )

        # Required skills receive the highest weight, while preferred
        # skills and technical depth provide supporting evidence.
        # Skill Score (25% weight in final features)
        features.skill_match_score = (
            0.50 * features.required_skills_match +
            0.20 * features.preferred_skills_match +
            0.15 * features.ai_technical_depth +
            0.15 * features.backend_technical_depth
        )

        # Measures how well the candidate's past work aligns with
        # the responsibilities and complexity of the target role.
        # Capability Score (20% weight in final features)
        features.capability_match_score = (
            0.40 * features.required_capabilities_match +
            0.30 * features.project_complexity_score +
            0.30 * features.role_consistency_score
        )

        # Industry familiarity is evaluated independently because
        # domain knowledge can reduce onboarding time.
        # Industry Score (10% weight in final features)
        features.industry_score = (
            get_industry_relevance_score(candidate.profile.current_industry)
        )

        # Combines institution quality, degree relevance,
        # and academic performance.
        # Education Score (15% weight in final features)
        features.education_score = (
            0.50 * features.education_tier_score +
            0.30 * features.degree_relevance_score +
            0.20 * features.education_grade_score
        )

        # Captures profile quality, recruiter engagement,
        # and communication responsiveness.
        # Recruiter Signals Score (10% weight in final features)
        features.recruiter_signal_score = (
            0.40 * features.profile_completeness_score +
            0.30 * features.recruiter_interest_score +
            0.30 * features.candidate_responsiveness_score
        )

        # Practical hiring constraints such as notice period
        # and work arrangement compatibility.
        # Logistics Alignment Score (10% weight in final features)
        logistics_score = (
            0.40 * features.notice_period_score +
            0.30 * features.work_mode_alignment_score +
            0.30 * features.career_growth_score
        )

        # Core candidate quality score before applying
        # logistics-related adjustments.
        # Calculate final aggregated features score
        features.final_score = (
            0.20 * features.experience_score +
            0.25 * features.skill_match_score +
            0.20 * features.capability_match_score +
            0.10 * features.industry_score +
            0.15 * features.education_score +
            0.10 * features.recruiter_signal_score
        )

        # Logistics factors influence ranking, but are intentionally
        # kept as a smaller adjustment than technical qualifications.
        # Add 10% logistics adjustment weight
        features.final_score = 0.90 * features.final_score + 0.10 * logistics_score

        return features

    # =====================================================================
    # Feature Extraction Helper Methods
    # =====================================================================

    def _experience_years_score(self, candidate: Candidate) -> float:
        exp = candidate.profile.years_of_experience

        # Normalize experience into a 0–1 range where
        # 15+ years receives maximum credit.
        return min(1.0, max(0.0, exp / 15.0)) # 15 years yields max score

    def _experience_fit_score(self, candidate: Candidate, jd: JobDescription) -> float:
        exp = candidate.profile.years_of_experience

        # Candidates inside the requested experience range
        # receive full credit.
        if jd.min_experience <= exp <= jd.max_experience:
            return 1.0

        # Penalize underqualified candidates proportionally
        # to the gap from the minimum requirement.
        if exp < jd.min_experience:
            return exp / jd.min_experience if jd.min_experience > 0 else 1.0

        # Over-qualified slightly discounted
        return max(0.5, jd.max_experience / exp if exp > 0 else 1.0)

    def _average_tenure_score(self, candidate: Candidate) -> float:
        stats = TimelineConsistencyAnalyzer.analyze(candidate)
        tenure = stats["average_tenure_months"]

        # Average tenure acts as a proxy for employment stability.
        return min(1.0, max(0.0, tenure / 36.0)) # 3 years average tenure yields max score

    def _career_stability_score(self, candidate: Candidate) -> float:
        stats = TimelineConsistencyAnalyzer.analyze(candidate)

        # Reuse stability analysis instead of relying solely
        # on the number of jobs in the candidate's history.
        return stats["stability_score"]

    def _required_skills_match(self, candidate: Candidate, jd: JobDescription) -> float:
        c_skills = {s.name.lower() for s in candidate.skills}

        if not jd.required_skills:
            return 1.0

        # Required skills are treated as hard requirements
        # and matched using exact skill names.
        matched = sum(1 for s in jd.required_skills if s.lower() in c_skills)

        return matched / len(jd.required_skills)

       
        # Preferred skills contribute to ranking but are not treated
        # as strict requirements like required skills.
        def _preferred_skills_match(self, candidate: Candidate, jd: JobDescription) -> float:
            c_skills = {s.name.lower() for s in candidate.skills}
        if not jd.preferred_skills:
            return 1.0

        matched = sum(1 for s in jd.preferred_skills if s.lower() in c_skills)
        return matched / len(jd.preferred_skills)

    def _ai_technical_depth(self, candidate: Candidate) -> float:
        c_skills = {s.name.lower() for s in candidate.skills}

        # Search both declared skills and work-history descriptions
        # because AI expertise is often demonstrated through projects.
        desc_text = " ".join([j.description.lower() for j in candidate.career_history])
        desc_text += " " + candidate.profile.summary.lower()

        ai_keywords = [
            "pytorch", "tensorflow", "jax", "transformers", "llm", "rag", "nlp", 
            "computer vision", "reinforcement learning", "machine learning", 
            "deep learning", "langchain", "llamaindex", "prompt engineering", 
            "huggingface", "vector search", "semantic search", "fine-tuning", 
            "peft", "lora", "genai", "generative ai"
        ]

        matches = sum(1 for kw in ai_keywords if kw in c_skills or kw in desc_text)

        # Three or more AI signals are considered sufficient
        # evidence of strong AI exposure.
        return min(1.0, max(0.0, matches / 3.0)) # 3+ AI signals yields full score

    def _backend_technical_depth(self, candidate: Candidate) -> float:
        c_skills = {s.name.lower() for s in candidate.skills}

        # Backend expertise is inferred from both technologies
        # and practical project experience.
        desc_text = " ".join([j.description.lower() for j in candidate.career_history])
        desc_text += " " + candidate.profile.summary.lower()

        backend_keywords = [
            "python", "fastapi", "flask", "django", "go", "rust", "c++", "java", 
            "spring boot", "node.js", "microservices", "rest api", "grpc", "docker", 
            "kubernetes", "aws", "gcp", "azure", "postgresql", "mysql", "redis", 
            "mongodb", "kafka", "rabbitmq", "system design", "scalability"
        ]

        matches = sum(1 for kw in backend_keywords if kw in c_skills or kw in desc_text)

        # Three or more backend indicators receive full credit.
        return min(1.0, max(0.0, matches / 3.0)) # 3+ backend signals yields full score

    def _required_capabilities_match(self, candidate: Candidate, jd: JobDescription) -> float:
        combined = (candidate.profile.summary + " " + " ".join([j.description for j in candidate.career_history])).lower()

        if not jd.required_capabilities:
            return 1.0

        # Capability matching relies on evidence present in
        # summaries and real project/work descriptions.
        matched = sum(1 for cap in jd.required_capabilities if cap.lower() in combined)

        return matched / len(jd.required_capabilities)

    def _project_complexity_score(self, candidate: Candidate) -> float:
        combined = (candidate.profile.summary + " " + " ".join([j.description for j in candidate.career_history])).lower()

        # Complexity-related keywords act as lightweight indicators
        # of large-scale systems and challenging engineering work.
        complexity_keywords = [
            "scale", "scalability", "optimiz", "performance", "latency", "throughput", 
            "distributed", "millions", "billions", "pipeline", "architect", 
            "large-scale", "concurrency", "async", "parallel", "profile", "refactor"
        ]

        matches = sum(combined.count(kw) for kw in complexity_keywords)

        return min(1.0, max(0.0, matches / 5.0)) # 5+ counts of complexity terms yields full score

    def _role_consistency_score(self, candidate: Candidate, jd: JobDescription) -> float:
        target_tokens = set(re.findall(r'\b\w+\b', jd.title.lower()))

        if not target_tokens or not candidate.career_history:
            return 1.0

        # Discard generic words
        # Remove broad role terms so matching focuses on specialization.
        generic = {"engineer", "developer", "programmer", "architect", "analyst", "senior", "lead", "staff", "principal", "junior", "associate", "intern"}

        target_role_tokens = target_tokens - generic

        if not target_role_tokens:
            target_role_tokens = target_tokens

        score_sum = 0.0

        for job in candidate.career_history:
            job_tokens = set(re.findall(r'\b\w+\b', job.title.lower()))
            overlap = len(job_tokens.intersection(target_role_tokens))

            if overlap > 0:
                score_sum += 1.0

        return score_sum / len(candidate.career_history)

    def _education_tier_score(self, candidate: Candidate) -> float:
        if not candidate.education:
            return 0.0

        tier = candidate.education[0].tier.lower()

        # Institution tier is used as a simple proxy
        # for educational competitiveness.
        if "tier_1" in tier:
            return 1.0

        if "tier_2" in tier:
            return 0.75

        if "tier_3" in tier:
            return 0.5

        return 0.25

    def _degree_relevance_score(self, candidate: Candidate) -> float:
        if not candidate.education:
            return 0.0

        edu = candidate.education[0]
        deg_field = (edu.degree + " " + edu.field_of_study).lower()

        relevance_keywords = [
            "computer science", "computer engineering", "information technology", 
            "software engineering", "artificial intelligence", "data science", 
            "mathematics", "statistics", "electrical engineering", "electronics"
        ]

        for kw in relevance_keywords:
            if kw in deg_field:
                return 1.0

        # General STEM fields
        # STEM backgrounds receive partial credit even when
        # they are not directly software-focused.
        stem_keywords = ["physics", "science", "engineering", "math"]

        for kw in stem_keywords:
            if kw in deg_field:
                return 0.7

        return 0.3

    def _education_grade_score(self, candidate: Candidate) -> float:
        if not candidate.education:
            return 0.0

        grade_str = candidate.education[0].grade.lower().strip()

        # Missing grade information receives a neutral score
        # instead of heavily penalizing the candidate.
        if not grade_str:
            return 0.7

        try:
            # Percentage matching
            pct_match = re.search(r'([\d.]+)\s*%', grade_str)

            if pct_match:
                val = float(pct_match.group(1))
                return max(0.0, min(1.0, val / 100.0))

            # Fraction matching (e.g. 8.5/10, 4.0/4.0)
            frac_match = re.search(r'([\d.]+)\s*/\s*([\d.]+)', grade_str)

            if frac_match:
                num = float(frac_match.group(1))
                denom = float(frac_match.group(2))

                if denom > 0:
                    return max(0.0, min(1.0, num / denom))

            # Plain number matching
            num_match = re.search(r'([\d.]+)', grade_str)

            if num_match:
                val = float(num_match.group(1))

                # Supports percentage, GPA, CGPA,
                # and 4-point grading systems.
                if val > 10.0:
                    return max(0.0, min(1.0, val / 100.0))
                elif val > 4.0:
                    return max(0.0, min(1.0, val / 10.0))
                else:
                    return max(0.0, min(1.0, val / 4.0))

        except Exception:
            pass

        return 0.7

    def _profile_completeness_score(self, candidate: Candidate) -> float:
        # Candidate data may contain either a single signal object
        # or a list depending on the parser/source format.
        # If signals is a list (based on Candidate definition but parse outputs single object)
        signals = candidate.signals

        if isinstance(signals, list):
            signal = signals[0] if signals else None
        else:
            signal = signals

        # Missing recruiter signals receive a neutral score
        # rather than being heavily penalized.
        if not signal:
            return 0.5

        return signal.profile_completeness_score / 100.0

    def _recruiter_interest_score(self, candidate: Candidate) -> float:
        signals = candidate.signals

        if isinstance(signals, list):
            signal = signals[0] if signals else None
        else:
            signal = signals

        if not signal:
            return 0.0

        # pyrefly: ignore [missing-import]
        import numpy as np

        # Log scaling prevents unusually high recruiter activity
        # from disproportionately affecting the final score.
        # Log scaling interest view count (cap at 100 views)
        views = min(1.0, np.log1p(signal.profile_views_received_30d) / np.log1p(100))

        # Log scaling appearances count (cap at 500 searches)
        apps = min(1.0, np.log1p(signal.search_appearance_30d) / np.log1p(500))

        # Log scaling saves count (cap at 20 saves)
        saves = min(1.0, np.log1p(signal.saved_by_recruiters_30d) / np.log1p(20))

        # Recruiter saves are weighted slightly higher because
        # they indicate stronger hiring intent than profile views.
        return 0.3 * views + 0.3 * apps + 0.4 * saves

    def _candidate_responsiveness_score(self, candidate: Candidate) -> float:
        signals = candidate.signals

        if isinstance(signals, list):
            signal = signals[0] if signals else None
        else:
            signal = signals

        if not signal:
            return 0.5

        rate = signal.recruiter_response_rate
        time_hours = signal.avg_response_time_hours

        # Fast response (within 24 hours) yields 1.0, degrading linearly to 48 hours
        time_score = max(0.0, min(1.0, 1.0 - (time_hours / 48.0)))

        # Balance response reliability and response speed equally.
        return 0.5 * rate + 0.5 * time_score

    def _notice_period_score(self, candidate: Candidate) -> float:
        signals = candidate.signals

        if isinstance(signals, list):
            signal = signals[0] if signals else None
        else:
            signal = signals

        if not signal:
            return 0.5

        days = signal.notice_period_days

        # Shorter notice periods increase hiring flexibility
        # and therefore receive higher scores.
        if days <= 15:
            return 1.0

        if days <= 30:
            return 0.8

        if days <= 60:
            return 0.5

        if days <= 90:
            return 0.2

        return 0.1

    def _work_mode_alignment_score(self, candidate: Candidate, jd: JobDescription) -> float:
        signals = candidate.signals

        if isinstance(signals, list):
            signal = signals[0] if signals else None
        else:
            signal = signals

        if not signal:
            return 0.5

        pref = signal.preferred_work_mode.lower()
        jd_mode = jd.work_mode.lower()

        # Flexible candidates are considered compatible
        # with any employer work arrangement.
        if pref == jd_mode or pref == "flexible":
            return 1.0

        if jd_mode == "remote" and pref in ["remote", "flexible"]:
            return 1.0

        # Willingness to relocate partially offsets
        # work mode mismatches.
        if signal.willing_to_relocate:
            return 0.8

        return 0.3

    def _career_growth_score(self, candidate: Candidate) -> float:
        if not candidate.career_history:
            return 0.5

        # Approximate career progression using title seniority
        # because detailed promotion history is unavailable.
        def get_title_seniority(title: str) -> int:
            t = title.lower()

            if "director" in t or "vp" in t or "vice president" in t:
                return 9

            if "manager" in t or "head" in t:
                return 8

            if "lead" in t or "principal" in t or "staff" in t:
                return 7

            if "senior" in t or "sr" in t:
                return 6

            if "engineer" in t or "developer" in t or "analyst" in t:
                return 4

            if "junior" in t or "jr" in t or "associate" in t:
                return 2

            if "intern" in t or "trainee" in t:
                return 1

            return 3

        # We assume history is listed chronologically. Compare current/latest with oldest role.
        # Scale seniority levels by company brand tiers to reward growth at prestigious firms
        latest_job = candidate.career_history[0]
        oldest_job = candidate.career_history[-1]

        # Company tier is included so growth at stronger organizations
        # contributes more to the progression signal.
        latest_seniority = get_title_seniority(latest_job.title) * get_company_tier_score(latest_job.company)
        oldest_seniority = get_title_seniority(oldest_job.title) * get_company_tier_score(oldest_job.company)

        if latest_seniority > oldest_seniority:
            return 1.0
        elif latest_seniority == oldest_seniority:
            return 0.6

        return 0.3