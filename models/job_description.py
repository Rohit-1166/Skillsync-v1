from dataclasses import dataclass, field


@dataclass
class JobDescription:

    raw_text: str = ""

    title: str = ""
    company: str = ""

    min_experience: float = 0.0
    max_experience: float = 100.0

    location: str = ""
    work_mode: str = ""
    employment_type: str = ""

    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    required_capabilities: list[str] = field(default_factory=list)

    responsibilities: list[str] = field(default_factory=list)

    hidden_expectations: list[str] = field(default_factory=list)

    preferred_background: list[str] = field(default_factory=list)
    rejected_background: list[str] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)