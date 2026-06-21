import csv
from reasoning.explanation_generator import ExplanationGenerator


# Produces the final competition-format submission CSV.
# Output schema is fixed — any column change breaks downstream evaluation.
# Delegates reasoning generation to ExplanationGenerator to keep
# formatting logic separate from narrative construction.
class SubmissionWriter:

    @staticmethod
    def write(
        ranked_candidates,
        output_file,
        jd,
        top_k=100
    ):
        """
        Writes the final matching results to a CSV file.
        Matches the exact schema required: candidate_id,rank,score,reasoning
        """
        # Re-sort before writing to guarantee rank order is determined
        # by score descending, with candidate_id as a tiebreaker for
        # deterministic output across runs with identical scores.
        sorted_candidates = sorted(
            ranked_candidates,
            key=lambda x: (-round(x[1].final_score, 4), x[0].candidate_id)
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            # Column order is mandated by the evaluation schema.
            # Do not reorder — downstream scoring depends on exact alignment.
            writer.writerow([
                "candidate_id",
                "rank",
                "score",
                "reasoning"
            ])

            for rank, (
                candidate,
                features
            ) in enumerate(
                sorted_candidates[:top_k],
                start=1
            ):
                # Reasoning is generated per candidate against the active JD
                # so the summary reflects the specific role context, not a
                # generic candidate description.
                reasoning = ExplanationGenerator.generate_csv_reasoning(
                    candidate=candidate,
                    features=features,
                    jd=jd
                )

                # Score is rounded to 4 decimal places to match the
                # precision used during ranking while keeping the CSV readable.
                writer.writerow([
                    candidate.candidate_id,
                    rank,
                    round(features.final_score, 4),
                    reasoning
                ])