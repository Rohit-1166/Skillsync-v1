import csv
from reasoning.explanation_generator import ExplanationGenerator

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

            # Header MUST match exactly: candidate_id,rank,score,reasoning
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
                # Generate dynamic factual reasoning summary for the candidate
                reasoning = ExplanationGenerator.generate_csv_reasoning(
                    candidate=candidate,
                    features=features,
                    jd=jd
                )

                writer.writerow([
                    candidate.candidate_id,
                    rank,
                    round(features.final_score, 4),
                    reasoning
                ])