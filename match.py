"""
match.py

Step 1 of the Upwork job-scoring agent.

What this script does:
1. Stores my list of skills.
2. Checks a job description to see which of my skills are mentioned in it.
3. Figures out which skills the JOB is asking for (not just how many of
   my skills I happen to know overall).
4. Turns that into a simple 0-100 "match score": how much of what the
   job needs do I actually have.
5. Prints the result for one example job description.

This is intentionally simple: it looks for each skill as a whole
word/phrase inside the job description (ignoring upper/lower case
differences), and makes sure overlapping skills (like "transformers"
inside "Hugging Face Transformers") only get counted once. Smarter
matching (synonyms, stemming, etc.) can be added later.
"""

# `re` is Python's built-in module for pattern matching (regular
# expressions). We use it below to match whole words/phrases instead of
# raw substrings.
import re

# -----------------------------------------------------------------------
# 1. MY SKILLS
# -----------------------------------------------------------------------
# This is just a plain Python list of strings. Each string is one skill
# I know. Later, this same list will be reused to build proposals.
MY_SKILLS = [
    "Python",
    "C++",
    "PyTorch",
    "TensorFlow",
    "Keras",
    "Hugging Face Transformers",
    "RAG",
    "scikit-learn",
    "OpenCV",
    "NumPy",
    "Pandas",
    "Git",
    "LLM fine-tuning",
    "PEFT/LoRA",
    "CNN",
    "ResNet",
    "NLP",
    "computer vision",
    "transformers",
]

# A small, hardcoded list of common ML keywords to check for in the job
# text, ON TOP OF my own skills list. This helps us guess what the job
# actually needs, even for terms that aren't (only) on my skill list.
# Some of these overlap with MY_SKILLS on purpose - that's fine, we
# combine both lists into a single set of unique requirements later.
COMMON_ML_KEYWORDS = [
    "Python",
    "PyTorch",
    "TensorFlow",
    "Hugging Face",
    "LoRA",
    "RAG",
    "NLP",
    "computer vision",
    "LLM",
]

# Some skills get matched under more than one name because MY_SKILLS and
# COMMON_ML_KEYWORDS use slightly different phrasing for the same thing
# (e.g. COMMON_ML_KEYWORDS has "LoRA" and "Hugging Face" as short forms,
# while MY_SKILLS has the fuller "PEFT/LoRA" and "Hugging Face
# Transformers"). Without this mapping, a single mention in the job text
# could be counted as two separate requirements.
#
# Keys must be lowercase (we always look them up in lowercase). Each key
# maps to the one "canonical" skill name we want to keep instead.
SKILL_ALIASES = {
    "lora": "PEFT/LoRA",
    "transformers": "Hugging Face Transformers",
    "hugging face": "Hugging Face Transformers",
}


def canonicalize_skills(skill_names):
    """
    Rewrite a list of skill names so that known aliases collapse into a
    single canonical name, and remove any duplicates that result.

    Example: ["PEFT/LoRA", "LoRA"] -> ["PEFT/LoRA"]
    (because SKILL_ALIASES maps "lora" to "PEFT/LoRA", so both entries
    end up being the same canonical name, and we only keep it once.)

    We keep the first-seen order of the input list, which keeps the
    printed output stable and easy to read.
    """
    canonical_names = []
    seen = set()

    for name in skill_names:
        # Look up the lowercased name in SKILL_ALIASES; if it's not a
        # known alias, just keep the name as-is.
        canonical_name = SKILL_ALIASES.get(name.lower(), name)

        if canonical_name not in seen:
            seen.add(canonical_name)
            canonical_names.append(canonical_name)

    return canonical_names


# -----------------------------------------------------------------------
# 2. FUNCTION: find which skills appear in a job description
# -----------------------------------------------------------------------
def find_matched_skills(job_description, skills):
    """
    Look through `job_description` and return a list of every skill
    (from `skills`) that appears somewhere in that text, as a WHOLE
    word or phrase (not buried inside a longer word), and without
    double-counting a shorter skill that is really just part of a
    longer skill match (e.g. "transformers" inside "Hugging Face
    Transformers").

    How it works, step by step:

    Step A - Find every possible match, with its position in the text.
        For each skill, we build a regex pattern that matches that
        skill only when it is NOT glued to a letter/digit on either
        side. We use `(?<![a-z0-9])` and `(?![a-z0-9])` for this
        instead of the regex "\\b" (word boundary), because "\\b"
        behaves oddly around skills that contain symbols like "C++"
        or "PEFT/LoRA" (their edges are punctuation, not letters).
        `re.finditer` gives us every occurrence's start/end position
        in the text, so "Git" cannot match inside "digital" (the "G"
        there is glued to "di" before it), and "RAG" cannot match
        inside "storage" or "average".

    Step B - Remove overlapping matches, keeping the longest one.
        We sort all the matches we found by length, longest first.
        Then we walk through them, and only keep a match if it does
        not overlap with a span (start/end position) we already kept.
        This is what makes "Hugging Face Transformers" "win" over the
        shorter "transformers" match sitting inside it.

    Step C - Return the surviving skills, in the same order as the
        original `skills` list (so the output is easy to read).
    """
    text_lower = job_description.lower()

    # Every match we find gets stored as (start_position, end_position, skill_name)
    candidate_matches = []

    for skill in skills:
        skill_lower = skill.lower()
        # re.escape() makes sure symbols in the skill (like "+" or "/")
        # are treated as literal characters, not special regex symbols.
        pattern = r"(?<![a-z0-9])" + re.escape(skill_lower) + r"(?![a-z0-9])"

        for match in re.finditer(pattern, text_lower):
            candidate_matches.append((match.start(), match.end(), skill))

    # Longest matches first, so they get first pick of the text they cover.
    candidate_matches.sort(key=lambda item: item[1] - item[0], reverse=True)

    accepted_spans = []  # list of (start, end) positions we've already claimed
    matched_skill_names = set()

    for start, end, skill in candidate_matches:
        # Does this candidate overlap any span we already accepted?
        overlaps_existing = any(
            start < accepted_end and end > accepted_start
            for accepted_start, accepted_end in accepted_spans
        )
        if not overlaps_existing:
            accepted_spans.append((start, end))
            matched_skill_names.add(skill)

    # Return matches in the same order as the original skills list.
    return [skill for skill in skills if skill in matched_skill_names]


# -----------------------------------------------------------------------
# 3. FUNCTION: figure out which skills THE JOB is asking for
# -----------------------------------------------------------------------
def find_job_requirements(job_description, my_skills, common_keywords):
    """
    Estimate the set of skills this specific job needs, so we can later
    score "how much of what the job needs do I have" instead of "how
    much of everything I know does this job need" (which unfairly
    punishes a perfect match just because my overall skill list is long).

    We estimate "what the job needs" as the UNION of:
      - any of MY_SKILLS that are mentioned in the job text, and
      - any of the COMMON_ML_KEYWORDS that are mentioned in the job text.

    Combining both matters because the common-keywords list can catch
    requirement phrasing that isn't identical to my own skill list
    (e.g. "Hugging Face" on its own, without the word "Transformers").
    We reuse find_matched_skills() for both lists since the whole-word
    matching logic is exactly the same either way.

    Because the two lists use different phrasing for some of the same
    skill (e.g. "LoRA" vs. "PEFT/LoRA"), we run the combined list through
    canonicalize_skills() before turning it into a set, so the same real
    mention doesn't get counted as two separate requirements.
    """
    requirements_from_my_skills = find_matched_skills(job_description, my_skills)
    requirements_from_common_keywords = find_matched_skills(job_description, common_keywords)

    # Combine both lists, then collapse known aliases (e.g. "LoRA" and
    # "PEFT/LoRA") into a single canonical name before deduplicating.
    all_requirements = requirements_from_my_skills + requirements_from_common_keywords
    return set(canonicalize_skills(all_requirements))


# -----------------------------------------------------------------------
# 4. FUNCTION: turn matched skills into a 0-100 score
# -----------------------------------------------------------------------
def compute_match_score(matched_skills, job_requirements):
    """
    Compute a percentage score based on how much of what THE JOB needs
    I actually have:

        score = (number of my skills matched / number of skills the job needs) * 100

    Example: if the job needs 5 skills and I match 4 of them, the score
    is (4 / 5) * 100 = 80. This is different from (and more meaningful
    than) dividing by the size of my entire skill list, because a job
    that only needs 3 skills - all of which I have - should score close
    to 100, not be capped low just because I know 19 skills overall.

    round(..., ) is used so we get a clean whole number like 80 instead
    of something like 80.000000004.
    """
    if not job_requirements:  # avoid dividing by zero if nothing was detected
        return 0

    score = (len(matched_skills) / len(job_requirements)) * 100
    return round(score)


# -----------------------------------------------------------------------
# 4. FUNCTION: turn the score into a plain-English summary
# -----------------------------------------------------------------------
def summarize_match(matched_skills, score):
    """
    Build a short, human-readable sentence describing how good the
    match is, using simple score thresholds:
        score >= 60  -> "strong fit"
        score >= 30  -> "partial fit"
        otherwise    -> "weak fit"

    These thresholds are just a starting point and can be tuned later
    once we see scores from real job postings.
    """
    if score >= 60:
        fit_level = "strong"
    elif score >= 30:
        fit_level = "partial"
    else:
        fit_level = "weak"

    if matched_skills:
        # Show at most 5 matched skills in the sentence, so it stays readable
        # even if many skills matched.
        highlighted_skills = ", ".join(matched_skills[:5])
        return (
            f"This looks like a {fit_level} fit. "
            f"Core skills matched: {highlighted_skills}."
        )
    else:
        return "This looks like a weak fit. None of my listed skills were found."


# -----------------------------------------------------------------------
# 5. FUNCTION: run the full pipeline on one job and print a report
# -----------------------------------------------------------------------
def run_match_report(job_title, job_description):
    """
    Run every step (matching, job-requirements detection, scoring,
    summary) on a single job description and print the results. Pulled
    out into its own function so we can reuse it for multiple example
    jobs without copy-pasting the same print statements each time.
    """
    print(f"===== {job_title} =====")

    # Step A: find which of my skills show up in this job description
    # (canonicalized too, so aliases collapse here as well, for consistency)
    matched = canonicalize_skills(find_matched_skills(job_description, MY_SKILLS))

    # Step B: figure out which skills the JOB is asking for
    job_requirements = find_job_requirements(job_description, MY_SKILLS, COMMON_ML_KEYWORDS)

    # Step C: compute the score as (my matches) / (what the job needs)
    score = compute_match_score(matched, job_requirements)

    # Step D: print the results so I can see them
    print("Matched skills:")
    for skill in matched:
        print(f"  - {skill}")

    print(f"\nJob requirements detected: {sorted(job_requirements)}")
    print(f"Match score: {score}/100")

    # Step E: print a plain-English summary of the result
    print(f"\n{summarize_match(matched, score)}")
    print()  # blank line to separate reports when running multiple jobs


# -----------------------------------------------------------------------
# 6. EXAMPLES: run the pipeline on sample job postings
# -----------------------------------------------------------------------
# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match.py`), not when
# it's imported by another file."
if __name__ == "__main__":
    llama_lora_job_description = """
    We are looking for a Machine Learning Engineer to help us build a
    Retrieval-Augmented Generation (RAG) pipeline for our internal
    knowledge base. You should be comfortable with Python, PyTorch, and
    Hugging Face Transformers. Experience with LLM fine-tuning and
    PEFT/LoRA is a big plus. Familiarity with Git for version control
    is required. Bonus points for NLP experience.
    """

    # A real Upwork job posting, used to sanity-check the matcher against
    # actual job text (not just a hand-written example).
    rag_document_qa_job_description = """
    AI Engineer needed to build RAG Document Q&A System

    I need an AI engineer to build a production-grade RAG system that allows
    users to upload PDF documents and ask questions in natural language,
    receiving accurate answers with source citations.
    Requirements:
    - Multi-format document ingestion (PDF, DOCX, TXT)
    - Arabic and English language support
    - Hybrid search: BM25 + semantic embeddings
    - Cross-encoder reranking for accuracy
    - Source citation with document name and page number
    - JWT authentication with multi-tenant support
    - Async background processing with job status tracking
    - Query caching
    - FastAPI REST API backend
    - Observability and tracing

    Mandatory skills: Artificial Intelligence, Machine Learning, Python,
    Artificial Neural Network
    """

    run_match_report("LLaMA / LoRA example job", llama_lora_job_description)
    run_match_report("Real Upwork job: RAG Document Q&A System", rag_document_qa_job_description)
