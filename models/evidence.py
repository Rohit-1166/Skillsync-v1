from dataclasses import dataclass, field


@dataclass
class Evidence:
    # Represents a piece of supporting evidence used to justify
    # capability detection and matching decisions.
    capability: str

    # Confidence score indicating how strongly the evidence
    # supports the identified capability.
    confidence: float

    # Records where the evidence was found
    # (e.g., skills, profile summary, career history).
    source: str

    # Stores the exact keywords or phrases that triggered
    # the capability match.
    matched_terms: list[str] = field(default_factory=list)