import os
# Force Hugging Face Hub to run in offline mode using the local cache
os.environ["HF_HUB_OFFLINE"] = "1"

from config.settings import JD_FILE
from utils.document_reader import DocumentReader
from parser.jd_parser import JDParser
from embeddings.embedding_pipeline import EmbeddingPipeline
from ranking.hybrid_ranker import HybridRanker
from submission.submission_writer import SubmissionWriter
from submission.debug_submission_writer import DebugSubmissionWriter
from reasoning.explanation_generator import ExplanationGenerator
from utils.logger import logger


def main():

    logger.info("Starting SkillSync matching pipeline...")

    # 1. Read and parse Job Description
    logger.info("Reading Job Description...")

    jd_text = DocumentReader.read(JD_FILE)

    jd = JDParser(jd_text).parse()

    logger.info(f"Parsed Job Description: '{jd.title}' at '{jd.company}'")

    # 2. Run Embedding Pipeline to load/generate candidate index
    pipeline = EmbeddingPipeline()

    faiss_index, candidates = pipeline.run()

    # 3. Perform hybrid ranking (semantic search + feature scoring)
    logger.info("Ranking candidates...")

    ranker = HybridRanker(pipeline.embedder, faiss_index)

    ranked = ranker.rank(candidates, jd)

    # 4. Write final submissions
    logger.info("Writing submission files...")

    SubmissionWriter.write(
        ranked,
        "submission.csv",
        jd,
        top_k=100
    )

    DebugSubmissionWriter.write(
        ranked,
        "debug_submission.csv",
        top_k=100
    )

    logger.info("Generating candidate explanations for top 100 candidates...")
    explanations_file = os.path.join("output", "candidate_explanations.md")
    
    with open(explanations_file, "w", encoding="utf-8") as f:
        f.write("# Candidate Explanations (Explainable AI Ranking)\n\n")
        f.write(f"This report lists recruiter-oriented explanations, score breakdowns, and concrete evidence tables for the top 100 candidates ranked against the Job Description: **{jd.title}** at **{jd.company}**.\n\n")
        f.write("---\n\n")
        
        for rank, (candidate, features) in enumerate(ranked[:100], start=1):
            explanation = ExplanationGenerator.generate(
                candidate=candidate,
                similarity=features.similarity_score,
                features=features,
                jd=jd
            )
            # Add rank header prefix
            explanation_with_rank = explanation.replace(
                f"## Candidate: {candidate.candidate_id}",
                f"## Rank #{rank}: Candidate {candidate.candidate_id}"
            )
            f.write(explanation_with_rank)

    logger.info(f"Candidate explanations written to '{explanations_file}'.")

    logger.info("Submission files written successfully.")

    logger.info("SkillSync pipeline completed successfully.")


if __name__ == "__main__":
    main()