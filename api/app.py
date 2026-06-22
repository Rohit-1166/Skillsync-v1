import os

# Force Hugging Face to use only locally cached models.
# Prevents model downloads during evaluation or deployment.
os.environ["HF_HUB_OFFLINE"] = "1"

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from config.settings import JD_FILE 
from parser.jd_parser import JDParser
from embeddings.embedding_pipeline import EmbeddingPipeline
from ranking.hybrid_ranker import HybridRanker
from reasoning.explanation_generator import ExplanationGenerator
from utils.document_reader import DocumentReader
from utils.logger import logger

# Startup state
pipeline = None
ranker = None
candidates = None
candidate_map = None
default_jd = None
current_jd = None

# TODO: add middleware for request logging/metrics if needed later

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, ranker, candidates, candidate_map, default_jd

    logger.info("Initializing SkillSync matching service...")

    try:
        pipeline = EmbeddingPipeline()
        faiss_index, candidates = pipeline.run()

        ranker = HybridRanker(pipeline.embedder, faiss_index)
        candidate_map = {c.candidate_id: c for c in candidates}

        # Load fallback JD if available
        if Path(JD_FILE).exists():
            jd_text = DocumentReader.read(JD_FILE)
            default_jd = JDParser(jd_text).parse()

            logger.info(
                f"Loaded default Job Description: '{default_jd.title}'"
            )
        else:
            logger.warning(
                f"Default Job Description file '{JD_FILE}' not found."
            )

        logger.info(
            "SkillSync matching service successfully loaded cache and index."
        )

    except Exception as e:
        logger.error(
            f"Failed to initialize SkillSync matching service: {e}"
        )

    yield

    logger.info("Shutting down SkillSync matching service...")

app = FastAPI(
    title="SkillSync AI Recruitment Engine API",
    description="Offline recruitment service with hybrid semantic-feature candidate ranking and explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# Request model for ranking candidates from raw JD text.
class RankTextRequest(BaseModel):
    jd_text: str
    top_k: int = 100
    semantic_weight: float = 0.60

@app.get("/", response_class=HTMLResponse)
def read_root():
    static_file = Path(__file__).parent / "static" / "index.html"

    if static_file.exists():
        return HTMLResponse(
            content=static_file.read_text(encoding="utf-8")
        )

    return HTMLResponse(
        content="<h1>SkillSync Dashboard File Not Found</h1>",
        status_code=404
    )

@app.get("/health")
def health_check():

    # If startup initialization failed, report service unavailable.
    if pipeline is None or ranker is None or candidates is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not initialized."
        )

    return {
        "status": "healthy",

        # Fallback model name is returned when the wrapper model
        # does not expose active_folding_model_name.
        "model": pipeline.embedder.model.active_folding_model_name
        if hasattr(
            pipeline.embedder.model,
            "active_folding_model_name"
        )
        else "BAAI/bge-small-en-v1.5",

        "total_candidates": len(candidates),
        "cache_loaded": True
    }

@app.post("/rank/text")
def rank_candidates_by_text(request: RankTextRequest):

    if ranker is None or candidates is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not initialized."
        )

    global current_jd

    try:
        jd = JDParser(request.jd_text).parse()

        # Store the latest JD so explanation endpoints
        # can reference the same ranking context.
        current_jd = jd

        # Retrieve a larger candidate pool before reranking.
        # This improves recall and usually produces better matches.
        ranked = ranker.rank(
            candidates,
            jd,
            top_k_retrieval=500,
            semantic_weight=request.semantic_weight
        )

        results = []

        for rank_idx, (candidate, features) in enumerate(
            ranked[:request.top_k],
            start=1
        ):
            results.append({
                "rank": rank_idx,
                "candidate_id": candidate.candidate_id,
                "final_hybrid_score": round(features.final_score, 4),
                "semantic_similarity_score": round(features.similarity_score, 4),
                "experience_years": candidate.profile.years_of_experience,
                "current_title": candidate.profile.current_title,
                "current_company": candidate.profile.current_company
            })

        return {
            "total_results": len(results),
            "candidates": results
        }

    except Exception as e:
        logger.error(f"Error ranking by text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank/pdf")
async def rank_candidates_by_pdf(
    file: UploadFile = File(...),
    top_k: int = Form(100),
    semantic_weight: float = Form(0.60)
):

    if ranker is None or candidates is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not initialized."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        global current_jd

        # Uploaded PDFs are temporarily stored on disk because
        # the document reader expects a file path as input.
        temp_dir = Path("cache") / "temp"
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / file.filename

        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from the uploaded JD and convert it
        # into the structured format used by the ranking pipeline.
        jd_text = DocumentReader.read(temp_file_path)
        jd = JDParser(jd_text).parse()

        current_jd = jd

        # Remove temporary file after successful extraction
        # to avoid unnecessary disk usage.
        os.remove(temp_file_path)

        ranked = ranker.rank(
            candidates,
            jd,
            top_k_retrieval=500,
            semantic_weight=semantic_weight
        )

        results = []

        for rank_idx, (candidate, features) in enumerate(
            ranked[:top_k],
            start=1
        ):
            results.append({
                "rank": rank_idx,
                "candidate_id": candidate.candidate_id,
                "final_hybrid_score": round(features.final_score, 4),
                "semantic_similarity_score": round(features.similarity_score, 4),
                "experience_years": candidate.profile.years_of_experience,
                "current_title": candidate.profile.current_title,
                "current_company": candidate.profile.current_company
            })

        return {
            "total_results": len(results),
            "candidates": results
        }

    except Exception as e:
        logger.error(f"Error ranking by PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/candidate/{candidate_id}/explain")
def explain_candidate(
    candidate_id: str,
    format: str = Query("json", pattern="^(json|markdown)$"),
    semantic_weight: float = Query(0.60)
):
    global current_jd

    if ranker is None or candidates is None or candidate_map is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not initialized."
        )

    # O(1) lookup using the precomputed candidate map.
    candidate = candidate_map.get(candidate_id)

    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {candidate_id} not found."
        )

    # Use the most recently ranked JD.
    # If no ranking request has been made, fall back to the default JD.
    active_jd = current_jd if current_jd is not None else default_jd

    if active_jd is None:
        raise HTTPException(
            status_code=500,
            detail="Job description not loaded."
        )

    try:
        # Recompute features against the active JD so the explanation
        # reflects the same scoring context used during ranking.
        features = ranker.feature_engineer.extract(
            candidate,
            active_jd
        )

        # Convert the JD into the semantic document format
        # expected by the embedding model.
        jd_doc = ranker._build_jd_document(active_jd)

        jd_embedding = ranker.embedder.encode(jd_doc)

        cand_doc = ExplanationGenerator._extract_evidence(
            candidate,
            active_jd
        )

        from embeddings.semantic_document_builder import SemanticDocumentBuilder

        # Build a unified candidate document containing
        # experience, skills, education and profile signals.
        cand_doc_str = SemanticDocumentBuilder.build(candidate)

        cand_embedding = ranker.embedder.encode(cand_doc_str)

        import numpy as np

        # Measure semantic alignment between JD and candidate profile.
        similarity = float(
            np.dot(jd_embedding, cand_embedding)
        )

        # Final score combines semantic relevance with
        # feature-engineered hiring signals.
        hybrid_score = (
            semantic_weight * similarity
            + (1.0 - semantic_weight) * features.final_score
        )

        features.final_score = hybrid_score
        features.similarity_score = similarity

        report = ExplanationGenerator.generate(
            candidate=candidate,
            similarity=similarity,
            features=features,
            jd=active_jd
        )

        if format == "markdown":
            return PlainTextResponse(report)

        # TODO: serialize full candidate profile instead of cherry-picking fields here
        details = {
            "headline": candidate.profile.headline,
            "summary": candidate.profile.summary,
            "location": candidate.profile.location,
            "country": candidate.profile.country,
            "years_of_experience": candidate.profile.years_of_experience,
            "current_title": candidate.profile.current_title,
            "current_company": candidate.profile.current_company,
            "current_industry": candidate.profile.current_industry,

            "skills": [
                {
                    "name": s.name,
                    "proficiency": s.proficiency,
                    "duration_months": s.duration_months
                }
                for s in candidate.skills
            ],

            "education": [
                {
                    "institution": e.institution,
                    "degree": e.degree,
                    "field_of_study": e.field_of_study,
                    "start_year": e.start_year,
                    "end_year": e.end_year,
                    "grade": e.grade
                }
                for e in candidate.education
            ],

            "career_history": [
                {
                    "company": c.company,
                    "title": c.title,
                    "duration_months": c.duration_months,
                    "description": c.description,
                    "start_date": c.start_date,
                    "end_date": c.end_date
                }
                for c in candidate.career_history
            ],

            "signals": {
                "notice_period_days": candidate.signals.notice_period_days,
                "preferred_work_mode": candidate.signals.preferred_work_mode,
                "willing_to_relocate": candidate.signals.willing_to_relocate,
                "profile_completeness_score":
                    candidate.signals.profile_completeness_score
            }
        }

        return {
            "candidate_id": candidate_id,
            "final_hybrid_score": round(hybrid_score, 4),
            "semantic_similarity_score": round(similarity, 4),

            # Human-readable recommendation derived from the final score.
            "recommendation_level":
                "STRONG MATCH" if hybrid_score >= 0.70
                else "GOOD MATCH" if hybrid_score >= 0.60
                else "MARGINAL MATCH" if hybrid_score >= 0.50
                else "UNALIGNED",

            "report_markdown": report,
            "candidate_details": details,
            "jd_required_skills": active_jd.required_skills if active_jd else [],
            "jd_title": active_jd.title if active_jd else "",
            "jd_company": active_jd.company if active_jd else ""
        }

    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/candidates/flagged")
def get_flagged_candidates():
    if ranker is None or candidates is None:
        raise HTTPException(
            status_code=503,
            detail="Matching engine is not initialized."
        )
    
    flagged = []
    for c in candidates:
        is_honeypot, reason = ranker._is_honeypot(c)
        if is_honeypot:
            flagged.append({
                "candidate_id": c.candidate_id,
                "current_title": c.profile.current_title or "N/A",
                "current_company": c.profile.current_company or "N/A",
                "reason": reason
            })
            
    return {
        "total_flagged": len(flagged),
        "flagged_candidates": flagged
    }