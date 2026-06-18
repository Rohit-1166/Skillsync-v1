# SkillSync — AI Recruitment Discovery & Ranking Engine

SkillSync is a production-grade, offline candidate discovery and hybrid ranking system built for the **India Runs Hackathon 2026 (AI Recruitment Intelligence Track)**. It is designed to match a massive pool of 100,000 candidates against a target Job Description with extreme accuracy, explainability, and speed, satisfying strict resource boundaries.

---

## 🚀 Key Differences & Architectural Innovations

Compared to standard keyword-matching or plain vector-embedding search engines, SkillSync introduces three major innovations:

1. **Strategic Honeypot & Trap Filters**: 
   - *The Trap*: Most candidate Discovery portals can be gamed by keyword stuffers (e.g., a candidate claiming "RAG / LLM" expertise whose actual title is "Marketing Manager").
   - *The Solution*: SkillSync parses and scores role consistency and career trajectory progression. If a candidate lists expert skills but has an unrelated title, or has impossible job intervals (such as 14 years tenure at a company that is only 3 years old), our system automatically detects and down-grades them.
   
2. **20 Advanced Recruiter-Inspired Features**:
   - Rather than relying on simple text similarity, SkillSync calculates 20 distinct features across 6 categories (Experience & Stability, Skill & Technical Depth, Capability & Alignment, Education Quality, Recruiter Engagement, and Logistics) to mimic the judgment of senior recruiting architects.

3. **Flat Cosine Semantic Caching (CPU Optimized)**:
   - High-throughput vector databases usually require GPUs. SkillSync utilizes a local, cached **FAISS index FlatIP** structure on CPU to fetch and score candidates.
   - Complete pipeline matching executes in **`0.45` seconds** for 100,000 candidates!

---

## 🛠️ Folder Structure & Architecture

```
SkillSync-v1/
├── api/
│   └── app.py                  # Production FastAPI service with matching & explainability endpoints
├── cache/                      # Flat cache for FAISS index and candidate embeddings
├── config/
│   ├── settings.py             # Global settings (model, paths, thresholds)
│   └── constants.py            # Global recruiter mappings (titles, degrees)
├── data/
│   ├── candidates.jsonl        # The raw 100,000-candidate pool
│   └── Job_Description.pdf     # The target Job Description document
├── embeddings/
│   ├── embedder.py             # SentenceTransformers wrapper
│   ├── embedding_pipeline.py   # Offline chunked indexing pipeline
│   └── semantic_document_builder.py # Natural language recruiter profile builder
├── features/
│   ├── consistency.py          # Career tenure & job-stability analyzer
│   └── feature_engineering.py  # 20 advanced recruiter feature scoring engine
├── knowledge/
│   ├── companies.py            # Tiered tech company brand database
│   ├── industries.py           # Industry relevance scoring dictionary
│   └── capabilities.py         # Recruiter skill-to-capability aliases
├── models/
│   ├── candidate.py            # Dataclasses for parsed profiles
│   ├── job_description.py      # Dataclasses for parsed JDs
│   ├── candidate_features.py   # Dataclasses for computed scores
│   └── evidence.py             # Dataclasses for matching evidence
├── output/
│   ├── candidate_explanations.md # Top-100 recruit-readable markdown reports
│   └── evaluation_report.md    # Performance profiling and score statistics
├── parser/
│   ├── candidate_parser.py     # Streaming JSONL reader and mapper
│   └── jd_parser.py            # Factual Job Description parser
├── ranking/
│   └── hybrid_ranker.py        # 60% semantic + 40% feature hybrid ranker (with honeypot filters)
├── submission/
│   ├── submission_writer.py    # Formatter for CSV submission
│   └── debug_submission_writer.py # Formatter for detailed CSV debug metrics
├── tests/                      # Python unittest suite
├── evaluation.py               # Profiling and statistics engine
├── main.py                     # Command-line entrypoint to run matcher pipeline
├── run_tests.py                # Automated unit test suite runner
├── submission.csv              # Monotonically ranked final output (validated)
├── debug_submission.csv        # Diagnostic spreadsheet with feature component columns
└── submission_metadata.yaml    # Hackathon portal submission metadata file
```

---

## 🔌 Setup & Local Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Rohit-1166/Skillsync-v1.git
   cd Skillsync-v1
   ```
2. Set up a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Reproduction & Running the Pipeline

To run the complete pipeline and produce the submission outputs:
```bash
python main.py
```
This single command:
1. Parses the default Job Description PDF (`data/Job_Description.pdf`).
2. Loads the parsed candidate database and checks for cached embeddings/FAISS index.
3. Automatically performs semantic search, filters honeypot candidates, and computes the 20 advanced recruiter features.
4. Generates the final [submission.csv](file:///c:/Users/iitbo/OneDrive/Desktop/SkillSync-v1/submission.csv) and [debug_submission.csv](file:///c:/Users/iitbo/OneDrive/Desktop/SkillSync-v1/debug_submission.csv) outputs in the root folder.
5. Generates the comprehensive candidate explanations report in [output/candidate_explanations.md](file:///c:/Users/iitbo/OneDrive/Desktop/SkillSync-v1/output/candidate_explanations.md).

---

## 🧪 Running the Test Suite

We use Python's built-in `unittest` framework. To run the automated test suite and check components:
```bash
python run_tests.py
```

---

## 📊 Evaluation & Latency Profile

To profile execution times, score distribution, and run feature correctness tests:
```bash
python evaluation.py
```
This produces the profiling report at [output/evaluation_report.md](file:///c:/Users/iitbo/OneDrive/Desktop/SkillSync-v1/output/evaluation_report.md) with details:
- **Total LATENCY**: **`0.4545 seconds`** for complete query matching on 100k candidates.
- **Honeypot Filter**: Flagged and skipped 67 impossible candidates in data stream.
- **Score Range**: Matches range from `0.55` to `0.77` with a standard deviation of `0.046`.
- **Normalization Bounds**: 🟢 100% PASS (all features lie strictly in `[0.0, 1.0]`).

---

## 🌐 FastAPI REST Service Usage

SkillSync provides a hosted API server to query candidate discoverability dynamically.

### Start the Server:
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

### Endpoints Available:
- **`GET /health`**: Returns system health status and cache metadata.
- **`POST /rank/text`**: Takes Job Description raw text and returns ranked candidate lists.
- **`POST /rank/pdf`**: Takes an uploaded Job Description PDF file and returns ranked candidate lists.
- **`GET /candidate/{candidate_id}/explain`**: Returns the recruiter explainability report in Markdown or JSON format.
  - Query parameters: `format=json` or `format=markdown`.
