from models.candidate import Candidate


class TimelineConsistencyAnalyzer:
    """
    Analyzes candidate career history for timeline consistency and tenure stability.
    Since career history entries only contain duration_months and is_current (no start/end dates),
    calendar gaps cannot be computed. Instead, stability is measured by tenure duration patterns.
    """

    @staticmethod
    def analyze(candidate: Candidate) -> dict:
        history = candidate.career_history
        if not history:
            return {
                "average_tenure_months": 0.0,
                "short_tenure_count": 0,
                "stability_score": 1.0,
                "total_duration_months": 0,
                "job_count": 0
            }

        total_months = sum(job.duration_months for job in history)
        job_count = len(history)
        average_tenure = total_months / job_count if job_count > 0 else 0.0

        short_tenure_count = 0
        for job in history:
            # Past jobs lasting less than 12 months indicate short-term hopping
            if not job.is_current and job.duration_months < 12:
                short_tenure_count += 1
            # Current jobs under 6 months are not penalized (new starts)
            elif job.is_current and job.duration_months < 6:
                pass

        stability_score = 1.0 - (short_tenure_count / job_count) if job_count > 0 else 1.0

        return {
            "average_tenure_months": average_tenure,
            "short_tenure_count": short_tenure_count,
            "stability_score": max(0.0, min(1.0, stability_score)),
            "total_duration_months": total_months,
            "job_count": job_count
        }
