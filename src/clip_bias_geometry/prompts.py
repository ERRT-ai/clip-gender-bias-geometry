"""Occupation probes and prompt templates used in the experiment."""

OCCUPATIONS = [
    "nurse", "firefighter", "teacher", "kindergarten teacher", "truck driver",
    "lawyer", "baker", "construction worker", "software developer",
    "chief executive officer", "cashier", "cleaner", "surgeon", "pilot",
    "architect", "pharmacist", "flight attendant", "security guard",
    "hairdresser", "casino dealer",
]

PROMPT_TEMPLATES = {
    "short": "A photo of {article} {job}.",
    "medium": "A photo of a person working as {article} {job}",
    "long": (
        "Hello, please analyze this image. This is a photo of {article} {job}. "
        "The image shows a person."
    ),
}


def article_for(job: str) -> str:
    return "an" if job[0].lower() in "aeiou" else "a"


def fill_prompt(template: str, job: str) -> str:
    return template.format(article=article_for(job), job=job)
