"""
Capability knowledge base.

Maps high-level recruiter capabilities to the various
skills, technologies and concepts that demonstrate them.
"""


CAPABILITY_MAP = {

    "retrieval": {

        "aliases": [
            "retrieval",
            "information retrieval",
            "search",
            "semantic search",
            "vector search",
            "hybrid search",
            "dense retrieval"
        ],

        "evidence": [
            "faiss",
            "milvus",
            "pinecone",
            "chromadb",
            "weaviate",
            "bm25",
            "elastic",
            "elasticsearch"
        ]
    },

    "embeddings": {

        "aliases": [
            "embedding",
            "embeddings"
        ],

        "evidence": [
            "sentence transformers",
            "bge",
            "e5",
            "bert",
            "embedding model"
        ]
    },

    "ranking": {

        "aliases": [
            "ranking",
            "reranking",
            "learning to rank"
        ],

        "evidence": [
            "recommendation",
            "recommender",
            "search ranking",
            "relevance",
            "ltr"
        ]
    },

    "llm": {

        "aliases": [
            "llm",
            "large language model",
            "foundation model"
        ],

        "evidence": [
            "rag",
            "langchain",
            "llamaindex",
            "prompt engineering",
            "agents",
            "lora",
            "peft",
            "fine tuning",
            "fine-tuning"
        ]
    },

    "backend": {

        "aliases": [
            "backend",
            "backend engineering"
        ],

        "evidence": [
            "python",
            "fastapi",
            "flask",
            "django",
            "rest api",
            "microservices",
            "docker",
            "kubernetes"
        ]
    }
}

def get_all_capabilities() -> list[str]:
    return list(CAPABILITY_MAP.keys())

def get_capability(capability: str) -> dict:
    return CAPABILITY_MAP.get(capability.lower(), {})
