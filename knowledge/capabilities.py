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

        # Technologies and tools that provide evidence
        # of retrieval and search-related experience.
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

        # Common embedding models and frameworks that
        # indicate experience with vector representations.
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

        # Keywords associated with ranking systems
        # and relevance optimization.
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

        # Terms commonly found in modern LLM
        # and Generative AI workflows.
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

        # Backend technologies used as indicators
        # of server-side engineering experience.
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
    # Returns all supported capability categories
    # available in the knowledge base.
    return list(CAPABILITY_MAP.keys())

def get_capability(capability: str) -> dict:
    # Case-insensitive lookup of capability metadata.
    return CAPABILITY_MAP.get(capability.lower(), {})