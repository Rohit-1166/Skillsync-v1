from dataclasses import dataclass, field


@dataclass
class JobDescription:

    # Original job description text before parsing.
    raw_text: str = ""

    # Core job information extracted from the JD.
    title: str = ""
    company: str = ""

    # Experience range expected for the role.
    min_experience: float = 0.0
    max_experience: float = 100.0

    # Location and employment preferences defined by the employer.
    location: str = ""
    work_mode: str = ""
    employment_type: str = ""

    # Skills explicitly required versus skills that are desirable
    # but not mandatory for the role.
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    # Higher-level capabilities inferred from the JD
    # such as retrieval, ranking, backend, or LLM expertise.
    required_capabilities: list[str] = field(default_factory=list)

    # Day-to-day responsibilities extracted from the posting.
    responsibilities: list[str] = field(default_factory=list)

    # Expectations that may not be directly stated as requirements
    # but can be inferred from the JD content.
    hidden_expectations: list[str] = field(default_factory=list)

    # Backgrounds that are preferred or discouraged
    # during candidate evaluation.
    preferred_background: list[str] = field(default_factory=list)
    rejected_background: list[str] = field(default_factory=list)

    # Flexible container for additional parsed information
    # that may be useful for future ranking enhancements.
    metadata: dict = field(default_factory=dict)