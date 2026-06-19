# Centralized constants used for validating preferred work arrangements
# extracted from candidate profiles and job descriptions.
SUPPORTED_WORK_MODES = {
    "remote",
    "onsite",
    "hybrid",
    "flexible"
}

# Degree types recognized by the matching and parsing pipeline.
# Keeping them in one place makes future updates easier.
SUPPORTED_DEGREES = {
    "B.Tech",
    "B.E.",
    "M.Tech",
    "M.E.",
    "M.S.",
    "PhD"
}

# Standardized seniority levels used when interpreting job titles
# and estimating candidate experience alignment.
SUPPORTED_SENIORITY_LEVELS = {
    "intern",
    "junior",
    "associate",
    "engineer",
    "senior",
    "staff",
    "principal",
    "lead",
    "manager",
    "director"
}