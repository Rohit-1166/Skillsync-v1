import re

from models.job_description import JobDescription
from knowledge.capabilities import CAPABILITY_MAP


class JDParser:

    def __init__(self, raw_text: str):

        self.raw_text = raw_text
        self.text = raw_text.lower()

    def parse(self) -> JobDescription:

        jd = JobDescription()

        jd.raw_text = self.raw_text

        jd.title = self._extract_title()
        jd.company = self._extract_company()

        jd.location = self._extract_location()
        jd.work_mode = self._extract_work_mode()
        jd.employment_type = self._extract_employment_type()

        (
            jd.min_experience,
            jd.max_experience
        ) = self._extract_experience()

        jd.required_skills = self._extract_required_skills()

        jd.required_capabilities = self._infer_capabilities(
            jd.required_skills
        )

        jd.hidden_expectations = self._extract_hidden_expectations()

        return jd

    def _extract_title(self):

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

        return 0.0, 100.0

    def _extract_required_skills(self):

        skills = []

        for capability in CAPABILITY_MAP.values():

            for word in capability["aliases"]:

                if word.lower() in self.text:
                    skills.append(word)

        return sorted(list(set(skills)))

    def _infer_capabilities(self, skills):

        capabilities = []

        for capability, data in CAPABILITY_MAP.items():

            for alias in data["aliases"]:

                if alias in skills:

                    capabilities.append(capability)

                    break

        return capabilities

    def _extract_hidden_expectations(self):

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