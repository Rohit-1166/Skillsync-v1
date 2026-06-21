import re

from models.job_description import JobDescription
from knowledge.capabilities import CAPABILITY_MAP


# Transforms raw Job Description text into a structured JobDescription object.
# Acts as the entry point for JD understanding before ranking begins.
# All downstream components — ranker, feature engineer, explainer —
# consume the structured output produced here.
class JDParser:

    def __init__(self, raw_text: str):

        self.raw_text = raw_text

        # Lowercased copy used for case-insensitive keyword matching
        # without mutating the original text needed for regex extraction.
        self.text = raw_text.lower()

    def parse(self) -> JobDescription:

        jd = JobDescription()

        # Preserve the original JD text so the explanation generator
        # can reference it directly without reconstructing it.
        jd.raw_text = self.raw_text

        jd.title = self._extract_title()
        jd.company = self._extract_company()

        jd.location = self._extract_location()
        jd.work_mode = self._extract_work_mode()
        jd.employment_type = self._extract_employment_type()

        # Experience range is unpacked as a tuple to allow
        # boundary comparisons during feature scoring.
        (
            jd.min_experience,
            jd.max_experience
        ) = self._extract_experience()

        jd.required_skills = self._extract_required_skills()

        # Capabilities are inferred from extracted skills rather than
        # parsed directly — maps surface keywords to semantic competency groups.
        jd.required_capabilities = self._infer_capabilities(
            jd.required_skills
        )

        # Hidden expectations capture soft signals not stated explicitly
        # but implied by culture and role language in the JD.
        jd.hidden_expectations = self._extract_hidden_expectations()

        return jd

    def _extract_title(self):

        # Targets the structured "Job Description:" header pattern
        # common in template-based JD formats.
        match = re.search(
            r"Job Description:\s*(.+)",
            self.raw_text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        return ""

    def _extract_company(self):

        match = re.search(
            r"Company:\s*(.+)",
            self.raw_text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        return ""

    def _extract_location(self):

        match = re.search(
            r"Location:\s*(.+)",
            self.raw_text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        return ""

    def _extract_work_mode(self):

        # Priority order matters — "hybrid" is checked first
        # because hybrid JDs often also contain the word "remote".
        if "hybrid" in self.text:
            return "Hybrid"

        if "remote" in self.text:
            return "Remote"

        if "onsite" in self.text:
            return "Onsite"

        return ""

    def _extract_employment_type(self):

        match = re.search(
            r"Employment Type:\s*(.+)",
            self.raw_text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

        return ""

    def _extract_experience(self):

        # Captures numeric ranges like "3-5 years" or "3 to 5 years".
        # \D+ handles varied separators between the two numbers.
        match = re.search(
            r"Experience Required:\s*(\d+)\D+(\d+)",
            self.raw_text,
            re.IGNORECASE
        )

        if match:

            return (
                float(match.group(1)),
                float(match.group(2))
            )

        # Default to an open-ended range when experience is not specified
        # so no candidates are incorrectly penalized during scoring.
        return 0.0, 100.0

    def _extract_required_skills(self):

        # Scans the JD against all known aliases in the CAPABILITY_MAP
        # to surface skills even when non-standard terms are used.
        skills = []

        for capability in CAPABILITY_MAP.values():

            for word in capability["aliases"]:

                if word.lower() in self.text:
                    skills.append(word)

        # Deduplication and sorting ensure consistent skill lists
        # across JDs for stable downstream comparisons.
        return sorted(list(set(skills)))

    def _infer_capabilities(self, skills):

        # Maps extracted skill aliases back to their parent capability group.
        # Breaks early per capability once any alias matches to avoid
        # adding the same capability multiple times.
        capabilities = []

        for capability, data in CAPABILITY_MAP.items():

            for alias in data["aliases"]:

                if alias in skills:

                    capabilities.append(capability)

                    break

        return capabilities

    def _extract_hidden_expectations(self):

        # Soft culture signals extracted from JD language that are
        # not listed as formal requirements but influence candidate fit.
        # These feed into the explanation generator's culture alignment section.
        expectations = []

        if "startup" in self.text:
            expectations.append("startup mindset")

        if "product" in self.text:
            expectations.append("product thinking")

        if "ship" in self.text:
            expectations.append("shipping mindset")

        if "scrappy" in self.text:
            expectations.append("ownership")

        return expectations