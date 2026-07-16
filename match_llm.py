"""
match_llm.py

Step 2 of the Upwork job-scoring agent: LLM-based scoring.

What this script does:
- Connects to a locally running Ollama server (see ask_ollama()).
- Has my profile hardcoded as a short text block (MY_PROFILE).
- Defines score_job(job_text), which sends BOTH my profile and a job
  posting to the model, asking it to act like a technical recruiter and
  return a score, strengths, gaps, and a verdict, in a fixed format.
- Prints the result nicely for one example job.

This is a different, LLM-based approach from match.py (which uses plain
keyword matching with no AI model involved). match.py is left untouched
as our simple, deterministic baseline to compare against.

Requirements before running this file:
1. Ollama must be installed and running on this machine.
2. The llama3.1:8b model must be pulled already (run: ollama pull llama3.1:8b).
   Ollama's default install runs a local web server at http://localhost:11434
   as soon as it starts, which is what this script talks to.
"""

# `requests` is a popular third-party library for making HTTP calls. We
# use it here to send a normal HTTP request to Ollama's local API,
# because Ollama exposes itself as a small web server on your machine.
import requests

# This is the standard local address Ollama listens on. "11434" is just
# the fixed port number Ollama always uses by default.
OLLAMA_URL = "http://localhost:11434/api/generate"

# The name of the model we want Ollama to use. This must match a model
# you've already pulled locally (check with `ollama list` in a terminal).
MODEL_NAME = "llama3.1:8b"

# My profile, as plain text. This gets pasted straight into the prompt
# we send the model, so it has context on my background when judging
# whether a job is a good fit.
MY_PROFILE = """[REDACTED - moved to gitignored config.json]"""

# Scoring rules for the model to follow. This has gone through two
# rounds of fixes:
# - v1 just asked for "a match score" with no guidance on how strict to
#   be, and the model came back too generous (82/100 on a job that was
#   mostly heavy backend work I can't do alone).
# - v2 added strict weighting of mandatory/core requirements, but
#   OVERCORRECTED: the model started penalizing gaps (e.g. "no
#   backend/infra experience") even on jobs that never asked for
#   backend/infra at all - it was judging me against my whole profile's
#   weaknesses instead of against what THIS job actually needs.
# - v3 (this version) explicitly forces the model to first extract what
#   the job actually requires, and only count a gap against me if it
#   maps to one of those requirements. A skill I lack that the job never
#   asked for must NOT lower the score.
STRICT_SCORING_GUIDELINES = """
Score as a STRICT freelance-fit judge, not an encouraging recruiter. Be
skeptical: most real jobs need several different skills, and partially
matching is not the same as being able to deliver the job. At the same
time, be FAIR: only judge the candidate against what THIS job actually
asks for, never against unrelated skills the candidate happens to lack.

Follow these steps before assigning a score:
1. Extract a short list of what THIS job actually requires, based only
   on the job posting text. Split that list into MANDATORY/CORE
   requirements (explicitly required/mandatory, or clearly central to
   what the job is about) versus NICE-TO-HAVE requirements (secondary
   features, bonus skills, or minor details).
2. Compare the candidate profile only against that extracted list.
   - A "gap" only counts if it maps to something on this list. If the
     job never mentioned or implied a skill (e.g. backend frameworks,
     deployment, a specific language), the candidate's lack of that
     skill is IRRELEVANT and must NOT be listed as a gap or lower the
     score.
   - If the candidate is missing one or more MANDATORY/CORE items from
     the extracted list, the score MUST drop sharply - a missing
     essential, job-relevant skill caps the score low even if many
     nice-to-haves match.
3. Only after core, job-relevant requirements are covered should
   nice-to-have matches push the score higher.

Use these score anchors as a guide:
- 80-100: the candidate covers essentially all of what THIS job actually
  requires (mandatory/core requirements included), so they could
  realistically deliver most of it alone.
- 40-60: the candidate covers only part of what THIS job requires -
  some job-relevant mandatory/core requirements are weak or missing.
- Below 40: the candidate is missing multiple job-relevant
  mandatory/core requirements and would need significant help to
  deliver this job.
"""

# The exact format we want the model to reply in. Spelling this out in
# the prompt (instead of just asking a free-form question) makes the
# model's reply predictable and easy to read - and easier to parse with
# code later, if we want to extract just the score, for example.
RESPONSE_FORMAT_INSTRUCTIONS = """
Respond in EXACTLY this format, with nothing before or after it, and no
markdown formatting (no asterisks, no headers). Fill in JOB REQUIRES
FIRST, and only list a gap in GAPS if it also appears in JOB REQUIRES -
if a candidate weakness is not in JOB REQUIRES, it must be left out of
GAPS entirely, since the job never asked for it:

JOB REQUIRES: <comma-separated list of what this job actually needs, mandatory items marked with (core)>
SCORE: <a single integer from 0 to 100>
STRENGTHS:
- <strength 1>
- <strength 2>
- <strength 3 (optional)>
GAPS:
- <gap - must be an item from JOB REQUIRES that the candidate lacks>
- <gap 2 (optional, same rule)>
VERDICT: <one short sentence>
"""


def ask_ollama(prompt):
    """
    Send `prompt` to the local Ollama server and return the model's
    text reply as a string.

    How it works:
    - Ollama's /api/generate endpoint expects a JSON body with at least
      "model" (which model to use) and "prompt" (what to ask it).
    - By default, Ollama STREAMS its reply back piece by piece. To keep
      this first test as simple as possible, we set "stream": False so
      Ollama waits and sends back the whole answer in one single
      response instead of many small chunks.
    - requests.post(...) sends the HTTP request; .json() parses the
      JSON reply into a normal Python dictionary.
    - The actual generated text lives under the "response" key of that
      dictionary, so that's what we return.
    """
    request_body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=request_body)

    # Raise an error early if Ollama responded with a failure status
    # code (e.g. model not found, server not running), instead of
    # silently continuing with a broken response.
    response.raise_for_status()

    response_data = response.json()
    return response_data["response"]


def score_job(job_text):
    """
    Ask the model to evaluate how well MY_PROFILE fits the given job
    posting (`job_text`), acting as a technical recruiter.

    We build one big prompt containing:
      1. Instructions telling the model what role to play (a strict
         freelance-fit judge, not an encouraging recruiter) and what to
         evaluate.
      2. The strict scoring rubric (see STRICT_SCORING_GUIDELINES
         above), so mandatory/core requirements are weighted far more
         heavily than nice-to-haves.
      3. My profile, so it knows my background.
      4. The job posting text, so it knows what's being asked for.
      5. The strict response format we want back (see
         RESPONSE_FORMAT_INSTRUCTIONS above), so the reply is
         consistent and easy to read every time.

    Returns the model's raw reply as a string (already in the format
    described above), which we can print directly.
    """
    prompt = f"""You are a strict freelance-fit judge deciding whether a candidate
should apply to a job posting. Your job is to protect the candidate's
time - be skeptical, not encouraging.

{STRICT_SCORING_GUIDELINES}

CANDIDATE PROFILE:
{MY_PROFILE}

JOB POSTING:
{job_text}

{RESPONSE_FORMAT_INSTRUCTIONS}
"""

    return ask_ollama(prompt)


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match_llm.py`), not
# when it's imported by another file."
if __name__ == "__main__":
    # The same real Upwork job used earlier to test match.py, so we can
    # compare the LLM-based score against the keyword-based one.
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

    print(f"Scoring job with {MODEL_NAME} via Ollama...\n")

    result = score_job(rag_document_qa_job_description)

    print("===== LLM Job Match Report: RAG Document Q&A System =====")
    print(result)
