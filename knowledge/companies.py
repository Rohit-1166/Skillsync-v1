import re


def get_company_tier_score(company_name: str) -> float:
    """
    Evaluate a company name and return a normalized prestige score.

    Scoring tiers:
    - 1.0: Tier 1 brands, including FAANG, leading AI/tech firms, and high-profile startups.
    - 0.7: Tier 2 brands, including prominent enterprise technology firms and major platform companies.
    - 0.4: Tier 3 brands, including global IT services, consultancies, and outsourcing firms.
    - 0.1: Unknown or other brands.
    """
    if not company_name:
        return 0.1

    # Standardize the incoming company name to simplify matching.
    name = company_name.lower().strip()

    # Remove common company suffixes and punctuation from the normalized name.
    name = re.sub(r'\b(inc|corp|co|ltd|limited|corporation|llc|pvt|gmbh|sa|plc)\b\.?', '', name).strip()
    name = re.sub(r'[^\w\s]', '', name)

    # Define brand groups by tier for name matching.
    tier_1 = {
        "google", "microsoft", "amazon", "meta", "netflix", "apple", "uber", "lyft",
        "stripe", "airbnb", "twitter", "spacex", "tesla", "nvidia", "snowflake",
        "databricks", "palantir", "salesforce", "openai", "redrob", "zoom", "tiktok", "bytedance"
    }

    tier_2 = {
        "adobe", "shopify", "spotify", "slack", "coinbase", "pinterest", "snap",
        "atlassian", "hubspot", "twilio", "dropbox", "github", "gitlab", "oracle", "ibm",
        "cisco", "intel", "qualcomm", "amd", "hewlett packard", "hp", "dell", "vmware", "yahoo",
        "walmart", "grab", "tinder", "okta", "datadog", "elastic", "mongodb"
    }

    tier_3 = {
        "tcs", "tata consultancy", "infosys", "wipro", "cognizant", "accenture",
        "capgemini", "hcl", "tech mahindra", "deloitte", "pwc", "ey", "kpmg",
        "wipro", "capgemini", "tata", "mahindra", "ltts", "lti", "mindtree"
    }

    # Match against each tier and return the corresponding score.
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
