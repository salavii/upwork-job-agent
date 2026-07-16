"""
match.py

Step 1 of the Upwork job-scoring agent.

What this script does:
1. Stores my list of skills.
2. Checks a job description to see which of my skills are mentioned in it.
3. Turns that into a simple 0-100 "match score".
4. Prints the result for one example job description.

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
# 3. FUNCTION: turn matched skills into a 0-100 score
# -----------------------------------------------------------------------
def compute_match_score(matched_skills, all_skills):
    """
    Compute a simple percentage score:

        score = (number of skills matched / total number of my skills) * 100

    Example: if I have 10 skills total and 3 of them were found in the
    job post, the score is (3 / 10) * 100 = 30.

    round(..., ) is used so we get a clean whole number like 30 instead
    of something like 30.000000004.
    """
    if not all_skills:  # avoid dividing by zero if the skill list is empty
        return 0

    score = (len(matched_skills) / len(all_skills)) * 100
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
# 5. EXAMPLE: run the functions above on one sample job posting
# -----------------------------------------------------------------------
# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match.py`), not when
# it's imported by another file."
if __name__ == "__main__":
    example_job_description = """
    We are looking for a Machine Learning Engineer to help us build a
    Retrieval-Augmented Generation (RAG) pipeline for our internal
    knowledge base. You should be comfortable with Python, PyTorch, and
    Hugging Face Transformers. Experience with LLM fine-tuning and
    PEFT/LoRA is a big plus. Familiarity with Git for version control
    is required. Bonus points for NLP experience.
    """

    # Step A: find which of my skills show up in this job description
    matched = find_matched_skills(example_job_description, MY_SKILLS)

    # Step B: compute the score based on how many skills matched
    score = compute_match_score(matched, MY_SKILLS)

    # Step C: print the results so I can see them
    print("Matched skills:")
    for skill in matched:
        print(f"  - {skill}")

    print(f"\nMatch score: {score}/100")

    # Step D: print a plain-English summary of the result
    print(f"\n{summarize_match(matched, score)}")
