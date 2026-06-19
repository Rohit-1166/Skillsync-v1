def get_industry_relevance_score(industry: str) -> float:
    """
    Evaluates how closely aligned a candidate's industry is with the target software/AI domain.
    - Core Tech / Software (1.0): Software development, Internet, IT, AI, tech.
    - Tech-Adjacent / STEM (0.7): Fintech, Telecom, Banking, Consulting, Healthtech, Edtech, etc.
    - Non-Tech / Others (0.3): Traditional sectors like manufacturing, retail, real estate, etc.
    """
    # Missing industry information receives the baseline score
    # rather than being treated as highly relevant.
    if not industry:
        return 0.3

    ind = industry.lower().strip()

    # Core Software/AI/Tech industries
    core_tech = {
        "software", "internet", "computer software", "information technology", "it", 
        "artificial intelligence", "data science", "tech", "technology", "saas", "cloud"
    }

    # Tech-adjacent / STEM / Quantitative industries
    tech_adjacent = {
        "telecom", "finance", "fintech", "banking", "semiconductor", "ecommerce", 
        "e-commerce", "hardware", "consulting", "healthtech", "edtech", "computer hardware",
        "engineering", "insurance", "aerospace", "defense", "biotech"
    }

    # Core technology domains receive maximum relevance because
    # they are most closely aligned with software and AI roles.
    # Check for core tech keywords
    for keyword in core_tech:
        if keyword in ind:
            return 1.0

    # Adjacent industries receive partial credit because they
    # often involve transferable technical skills and workflows.
    # Check for tech-adjacent keywords
    for keyword in tech_adjacent:
        if keyword in ind:
            return 0.7

    # Default score for industries with limited direct overlap
    # with software engineering and AI-focused roles.
    return 0.3