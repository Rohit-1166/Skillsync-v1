from dataclasses import dataclass, field


@dataclass
class Evidence:
    capability: str
    confidence: float
    source: str
    matched_terms: list[str] = field(default_factory=list)