import re


def get_company_tier_score(company_name: str) -> float:
    """
    Evaluates a company's brand prestige tier and returns a score.
    - Tier 1 (1.0): FAANG + top AI & tech companies + top startups.
    - Tier 2 (0.7): High-tier enterprise tech + prominent tech giants.
    - Tier 3 (0.4): Global IT services, consultancies, and outsourcing companies.
    - Unknown/Others (0.1).
    """
    if not company_name:
        return 0.1

    name = company_name.lower().strip()

    # Normalize suffixes and punctuation
    name = re.sub(r'\b(inc|corp|co|ltd|limited|corporation|llc|pvt|gmbh|sa|plc)\b\.?', '', name).strip()
    name = re.sub(r'[^\w\s]', '', name) # Remove special characters

    # Tier 1 (FAANG / Top Tier Tech & Startups)
    tier_1 = {
        "google", "microsoft", "amazon", "meta", "netflix", "apple", "uber", "lyft", 
        "stripe", "airbnb", "twitter", "spacex", "tesla", "nvidia", "snowflake", 
        "databricks", "palantir", "salesforce", "openai", "redrob", "zoom", "tiktok", "bytedance"
    }

    # Tier 2 (Prominent Tech Giants / Elite Enterprise)
    tier_2 = {
        "adobe", "shopify", "spotify", "slack", "coinbase", "pinterest", "snap", 
        "atlassian", "hubspot", "twilio", "dropbox", "github", "gitlab", "oracle", "ibm", 
        "cisco", "intel", "qualcomm", "amd", "hewlett packard", "hp", "dell", "vmware", "yahoo",
        "walmart", "grab", "tinder", "okta", "datadog", "elastic", "mongodb"
    }

    # Tier 3 (Global IT Services / Consultancy / Outsourcing)
    tier_3 = {
        "tcs", "tata consultancy", "infosys", "wipro", "cognizant", "accenture", 
        "capgemini", "hcl", "tech mahindra", "deloitte", "pwc", "ey", "kpmg",
        "wipro", "capgemini", "tata", "mahindra", "ltts", "lti", "mindtree"
    }

    # Match based on word boundaries or substrings
    for brand in tier_1:
        if brand in name or name in brand:
            return 1.0

    for brand in tier_2:
        if brand in name or name in brand:
            return 0.7

    for brand in tier_3:
        if brand in name or name in brand:
            return 0.4

    return 0.1
