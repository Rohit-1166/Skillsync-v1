from models.candidate import Candidate


class SemanticDocumentBuilder:
    """
    Builds a recruiter-oriented semantic document for embedding.

    The generated document is natural language instead of a raw JSON dump.
    This improves embedding quality considerably.
    """

    @staticmethod
    def build(candidate: Candidate) -> str:

        profile = candidate.profile

        lines = []

        # =====================================================
        # Candidate Overview
        # =====================================================

        lines.append(
            f"The candidate currently works as "
            f"{profile.current_title} "
            f"at {profile.current_company}."
        )

        lines.append(
            f"The candidate has "
            f"{profile.years_of_experience:.1f} years "
            f"of professional experience."
        )

        if profile.current_industry:
            lines.append(
                f"The candidate works in the "
                f"{profile.current_industry} industry."
            )

        if profile.summary.strip():

            lines.append(
                f"Professional Summary: "
                f"{profile.summary.strip()}"
            )

        # =====================================================
        # Skills
        # =====================================================

        if candidate.skills:

            # Prioritize skills with stronger evidence by considering
            # endorsements and duration of usage before embedding.
            skills = sorted(
                candidate.skills,
                key=lambda s: (
                    s.endorsements,
                    s.duration_months
                ),
                reverse=True
            )

            skill_sentence = []

            for skill in skills:

                skill_sentence.append(

                    f"{skill.name}"

                )

            lines.append(

                "Technical Skills: "

                + ", ".join(skill_sentence)

                + "."

            )

        # =====================================================
        # Career
        # =====================================================

        if candidate.career_history:

            lines.append(

                "Professional Experience:"

            )

            for job in candidate.career_history:

                # Convert structured career data into readable
                # recruiter-style sentences for better semantic matching.
                paragraph = (

                    f"The candidate worked as "

                    f"{job.title} "

                    f"at {job.company}. "

                    f"The company operates in "

                    f"{job.industry}. "

                    f"The role lasted "

                    f"{job.duration_months} months. "

                    f"{job.description}"

                )

                lines.append(paragraph)

        # =====================================================
        # Education
        # =====================================================

        if candidate.education:

            # Assumes the parser keeps the most relevant or highest
            # qualification at index 0.
            education = candidate.education[0]

            lines.append(

                f"The candidate completed "

                f"{education.degree} "

                f"in "

                f"{education.field_of_study} "

                f"from "

                f"{education.institution}."

            )

        # =====================================================
        # Recruiter Signals
        # =====================================================

        signal = candidate.signals

        if signal:

            if signal.open_to_work_flag:

                lines.append(

                    "The candidate is actively open to work."

                )

            # Fast availability is often a useful ranking signal
            # for recruiters and hiring teams.
            if signal.notice_period_days <= 30:

                lines.append(

                    "The candidate can join quickly."

                )

            if signal.github_activity_score > 0:

                lines.append(

                    f"GitHub activity score is "

                    f"{signal.github_activity_score:.1f}."

                )

        # Join sections with blank lines to create a more natural
        # document structure for the embedding model.
        return "\n\n".join(lines)