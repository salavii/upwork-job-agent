"""
match_llm.py

Step 2 of the Upwork job-scoring agent: LLM-based scoring.

What this script does:
- Connects to a locally running Ollama server (see ask_ollama()).
- Loads your profile from config.json (see load_profile_config()).
- Defines score_job(job_text), which sends BOTH your profile and a job
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

# `re` (regular expressions) is used to pull just the SCORE and VERDICT
# lines out of the model's reply, so we can rank jobs without printing
# the model's full raw text for each one.
import re

# `html` (standard library) escapes text like "<" and "&" into their
# safe HTML equivalents, so job text or model output can never
# accidentally break our generated HTML page's structure.
import html

# `datetime` is used to stamp today's date onto the HTML report title,
# and to turn each job's "date_added" string into a nicely formatted
# section heading (e.g. "July 17, 2026").
import datetime

# `json` is used to read jobs.json, which is now written by server.py as
# a JSON list of job objects (see server.py's save_jobs()).
import json

# `sys` gives us access to command-line arguments (sys.argv), so this
# file can be run as `python match_llm.py --list`, `--delete N`, or
# `--fresh` instead of only the default scoring run.
import sys

# `os` is used to read cloud LLM API keys from environment variables
# (ANTHROPIC_API_KEY / OPENAI_API_KEY) - never from config.json or
# anywhere else in the codebase, so a key can never end up committed.
import os

# `hashlib` is used to turn a job's text into a short, stable ID (a
# hash), so we can cache that job's result in results.json and recognize
# "this is the same job I already scored" even without needing a
# database - two identical job texts always hash to the same value.
import hashlib

# sources.py is the new, automatic job-discovery module (PART A/B/C of
# this feature): it pulls jobs from public job-board APIs (RemoteOK,
# Remotive), normalizes them into the same shape jobs.json already uses,
# and filters/dedups them. Importing it here does not touch the
# extension + server.py flow in any way - that flow keeps writing to
# jobs.json exactly as before; --fetch/--daily just also read/write the
# same file.
import sources

# The file server.py saves scraped jobs into: a JSON list of objects
# shaped like {"text": "...", "date_added": "YYYY-MM-DD", "saved_at": "..."}.
JOBS_JSON_FILE = "jobs.json"

# Where already-scored jobs are cached, keyed by a hash of their text, so
# re-running this script doesn't re-call Ollama for jobs we've already
# scored. See load_results()/save_results()/hash_job_text() below.
RESULTS_JSON_FILE = "results.json"

# Where the HTML report gets written.
REPORT_FILE = "report.html"

# Where --daily's dedicated digest gets written (separate from the full
# report.html, which still shows every job, scored or not). This is
# rendered directly from JOB_LIST_FILE - see update_persistent_job_list()
# and rebuild_daily_report() below.
DAILY_REPORT_FILE = "daily_report.html"

# The PERSISTENT job list --daily accumulates into, across every run
# forever - unlike report.html/daily_report.html (which are just
# generated views), this JSON file IS the durable state:
#   {
#     "jobs": {"<job hash>": {"title", "score", "verdict", "flags",
#                              "source", "url", "date_posted", "date_found"}, ...},
#     "removed_hashes": ["<job hash>", ...]
#   }
# A job earns a permanent spot here once (see update_persistent_job_list())
# and stays until you remove it via daily_report.html's per-card "Remove"
# button (server.py's /remove_daily_job) or the "Clear all" button
# (/clear_daily_list) - a removed job's hash goes into "removed_hashes"
# so a later run can never silently re-add it.
JOB_LIST_FILE = "job_list.json"

# Only automatically-sourced jobs scoring ABOVE this make it into the
# persistent list at all - this is the "auto-drop bad jobs" bar. Chosen
# to sit comfortably above DOMAIN_MISMATCH_SCORE_CAP (15) so a role-type
# mismatch can never sneak in just because of scoring noise.
FIT_SCORE_THRESHOLD = 40

# Only jobs scoring at or above this get a drafted proposal - low-scoring
# jobs aren't worth spending the client's (or the model's) time on.
PROPOSAL_SCORE_THRESHOLD = 70

# Skill-gap analysis (per-job "Strong match, but missing: ..." line and
# the aggregate "focus on these skills" summary - see
# compute_skill_gap_summary()) only runs on jobs scoring ABOVE this -
# these are the realistic near-misses worth analyzing; a job scoring 20
# was never in reach regardless of which specific skills it's missing,
# so listing its gaps would just be noise.
SKILL_GAP_SCORE_THRESHOLD = 60

# ============================================================
# CODE-LEVEL scoring gates.
#
# Testing repeatedly showed the local 8B model's own self-reported
# judgment (DOMAIN FIT, its COMPONENTS fraction) is not reliable enough
# to trust alone: business/ops/sales roles that merely mention "AI" in
# passing scored 75-100/100 because the model conflates "this text talks
# a lot about AI" with "this role IS AI engineering". Rather than keep
# patching the prompt and hoping, ROLE TYPE is a deterministic, HARD cap
# in code, immune to the model's own reasoning.
#
# SENIORITY and YEARS-OF-EXPERIENCE, by contrast, are deliberately SOFT
# penalties, not caps: this is a free job board with no per-application
# cost (unlike Upwork), so a senior/high-experience ML role that
# otherwise matches the resume well should still surface, just ranked a
# bit lower - the user wants to see it and decide for themselves, not
# have the tool hide it.
#
# LOCATION/ELIGIBILITY has NO score effect at all - it's purely an
# informational FLAG (see location_flag()) shown in the report, since
# the user wants to see and judge these themselves too.
# ============================================================

# Gate 0 (existing): the model's own DOMAIN FIT self-report.
DOMAIN_MISMATCH_SCORE_CAP = 15

# Gate 1: ROLE TYPE (HARD cap - unchanged). Only hands-on ML/AI/
# data-science engineering or developer roles are what this profile is
# for - a title containing any of these words indicates a different
# FUNCTION (leadership, sales, recruiting, admin, ...) regardless of how
# much AI vocabulary surrounds it, and is capped at
# DOMAIN_MISMATCH_SCORE_CAP. Deliberately no "but it also says engineer"
# exception for management/director/head of/principal/staff -
# "Engineering Manager" or "Director of AI" are still leadership
# functions, not hands-on IC work.
ROLE_TYPE_MISMATCH_PATTERN = re.compile(
    r"\b(management|manager|director|head of|principal|staff|strategy"
    r"|product manager|sales|business development|recruiter|recruiting"
    r"|accounting|accountant|finance|communications?|marketing"
    r"|subject matter expert|executive assistant|analyst|it support"
    r"|frontend|front-end|video editor|data annotator|consultant"
    r"|customer (success|solutions|support)"
    r"|value engineer|solutions? engineer|sales engineer|manufacturing"
    r"|account|pre-sales)\b",
    re.IGNORECASE,
)

# Backstop for Gate 1: some non-engineering roles don't signal it in the
# TITLE at all (e.g. an "AI Technology Expert" title that reads as a
# hands-on role but whose actual responsibilities are business-process/
# consulting/customer-facing work). Checked against the job's full TEXT,
# not just the title - but only when the title does NOT already contain
# a clear engineering word (engineer, scientist, developer, researcher),
# so a real "Senior ML Engineer" posting that happens to mention
# "customer-facing" once in a stakeholder-communication bullet isn't
# capped just for that.
ENGINEERING_TITLE_PATTERN = re.compile(
    r"\b(engineer(ing)?|scientist|developer|research(er)?)\b", re.IGNORECASE
)
BUSINESS_PROCESS_CONTENT_PATTERN = re.compile(
    r"\b(business process(es)?|stakeholder management|customer-facing"
    r"|solution consulting|value engineering|sales cycle"
    r"|account management|pre-sales|post-sales|manufacturing operations"
    r"|process improvement)\b",
    re.IGNORECASE,
)

# Gate 2: SENIORITY (SOFT penalty, not a cap). This profile is
# early-career (M.Sc. student, ~1-2 years hands-on) - a title signaling
# heavy seniority, or a stated requirement of more than
# SENIORITY_MAX_YEARS years of professional experience, gets a flat
# points deduction instead of being capped, so a strong-fitting senior
# role still ranks reasonably rather than getting buried.
SENIORITY_TITLE_PATTERN = re.compile(r"\b(senior|lead)\b", re.IGNORECASE)
YEARS_EXPERIENCE_PATTERN = re.compile(
    r"(\d{1,2})\+?\s*(?:-\s*\d{1,2}\s*)?\s*years?\s+(?:of\s+)?(?:professional\s+|relevant\s+)?experience",
    re.IGNORECASE,
)
SENIORITY_MAX_YEARS = 4
SENIORITY_PENALTY_POINTS = 12

# Gate 3: LOCATION/ELIGIBILITY (INFORMATIONAL FLAG ONLY, no score
# effect - that's the user's call to make, not the tool's to hide). Split
# into two patterns because they mean different things for a freelancer
# working as a CONTRACTOR from Italy:
# - ON-SITE/citizenship/clearance requirements ALWAYS apply, regardless
#   of contract vs. employee - physical presence or citizenship isn't
#   something a contract status changes.
# - "US-based"/"work authorization" language is normally about W2
#   EMPLOYMENT eligibility - a contractor can legally do freelance/
#   contract/part-time work for a US (or any) client while physically
#   based in Italy (see config.json's "work_eligibility"), so this ONLY
#   gets flagged for postings that read as a full-time EMPLOYEE role
#   (see is_full_time_employee_role() below) - see location_flag().
# This is a heuristic text search, not true NLP - it can't detect
# negation (e.g. "no work authorization required" would still match),
# so treat the flag as a prompt to go check, not gospel.
ON_SITE_RESTRICTION_PATTERN = re.compile(
    r"\b(on-?site|in-office|in office"
    r"|must be (located|based|residing) in|must reside in"
    r"|citizens? only|security clearance|relocat(e|ion) to)\b",
    re.IGNORECASE,
)
EMPLOYEE_ONLY_RESTRICTION_PATTERN = re.compile(
    r"\b(us[\s-]based|u\.s\.[\s-]based|united states[\s-]based"
    r"|work authorization|authorized to work in)\b",
    re.IGNORECASE,
)

# Gate 5: JOB TYPE (HARD EXCLUSION, config-driven - see WORK_ELIGIBILITY
# below). A posting explicitly offering contract/freelance/part-time
# work always wins over an ambiguous or absent employment-type mention;
# anything else that mentions "full-time" is treated as a permanent
# EMPLOYEE role. If config.json's work_eligibility.full_time_employee_ok
# is false, a job identified as full-time-employee is EXCLUDED ENTIRELY
# from the persistent list (see update_persistent_job_list()) - not
# penalized, not flagged, just never surfaced, since it's not a role
# this profile is legally eligible for at all. The default (missing
# work_eligibility section, or full_time_employee_ok left true) is a
# complete no-op, so this never affects anyone without this constraint.
CONTRACT_TYPE_PATTERN = re.compile(
    r"\b(contract|contractor|freelance|part[\s-]?time|1099"
    r"|independent contractor|c2c|corp-to-corp)\b",
    re.IGNORECASE,
)
FULL_TIME_PATTERN = re.compile(r"\bfull[\s-]?time\b", re.IGNORECASE)

# Gate 4: DEAD POSTINGS. A posting that says there's no open role right
# now isn't a job at all - see sources.py's is_dead_posting() for where
# these get filtered out at FETCH time (so they never reach jobs.json in
# the first place); DEAD_POSTING_PATTERN is defined here so match_llm.py
# doesn't need to import sources.py just for this one regex, and reused
# below to catch anything that slipped in before that filter existed.
DEAD_POSTING_PATTERN = re.compile(
    r"don'?t currently have any open (roles?|positions?)"
    r"|no (current|open) (openings|positions|roles)"
    r"|not currently hiring",
    re.IGNORECASE,
)

# The browser extension saves manual Upwork jobs with the title formatted
# as "<actual title> - <Upwork category>" (e.g. "AI/ML Engineer Needed
# for a Simple AI Assistant (MVP) - Digital Marketing" - the category is
# Upwork's OWN classification, not part of the real job title. Checking
# ROLE_TYPE_MISMATCH_PATTERN/SENIORITY_TITLE_PATTERN against the whole
# string caused false caps (a real ML job filed under Upwork's "Digital
# Marketing" category got capped to 15 purely because "marketing"
# appeared in the tag). Only strip a trailing " - <category>" when it
# matches a known Upwork category name, so a job whose REAL title
# legitimately ends in " - Something" (e.g. auto-sourced job titles
# never have this pattern) is never mangled.
KNOWN_UPWORK_CATEGORIES = {
    "AI & Machine Learning", "Web Development", "Data Analysis & Testing",
    "Digital Marketing", "DevOps & Solution Architecture",
    "Lead Generation & Telemarketing", "Design & Creative", "Writing",
    "Admin Support", "Customer Service", "Sales & Marketing",
    "Accounting & Consulting", "Legal", "Engineering & Architecture",
    "Translation", "IT & Networking", "Data Science & Analytics",
}


def strip_upwork_category_suffix(title):
    """See KNOWN_UPWORK_CATEGORIES above for why this exists."""
    match = re.search(r"^(.*) - ([^-]+)$", title)
    if match and match.group(2).strip() in KNOWN_UPWORK_CATEGORIES:
        return match.group(1).strip()
    return title


def role_type_mismatch(title):
    """Gate 1 - see ROLE_TYPE_MISMATCH_PATTERN above."""
    return bool(ROLE_TYPE_MISMATCH_PATTERN.search(strip_upwork_category_suffix(title)))


def content_suggests_non_engineering_role(title, job_text):
    """Gate 1 backstop - see BUSINESS_PROCESS_CONTENT_PATTERN above."""
    if ENGINEERING_TITLE_PATTERN.search(strip_upwork_category_suffix(title)):
        return False
    return bool(BUSINESS_PROCESS_CONTENT_PATTERN.search(job_text))


def seniority_mismatch(title, job_text):
    """Gate 2 - see SENIORITY_TITLE_PATTERN/YEARS_EXPERIENCE_PATTERN above."""
    if SENIORITY_TITLE_PATTERN.search(strip_upwork_category_suffix(title)):
        return True
    for match in YEARS_EXPERIENCE_PATTERN.finditer(job_text):
        if int(match.group(1)) > SENIORITY_MAX_YEARS:
            return True
    return False


def location_flag(job_text):
    """
    Gate 3 - see ON_SITE_RESTRICTION_PATTERN/EMPLOYEE_ONLY_RESTRICTION_PATTERN
    above. Returns an informational flag string, or None if nothing
    applies. NEVER affects the score - see apply_score_adjustments().
    """
    on_site_match = ON_SITE_RESTRICTION_PATTERN.search(job_text)
    if on_site_match:
        return (
            f'Possible location restriction: "{on_site_match.group(0)}" '
            "- requires physical presence/citizenship, check if you qualify."
        )

    # "US-based"/"work authorization" language is about EMPLOYEE
    # eligibility - irrelevant to a contractor, so skip it entirely for
    # postings that read as contract/freelance/part-time work.
    if is_contract_type_role(job_text) and not is_full_time_employee_role(job_text):
        return None

    employee_match = EMPLOYEE_ONLY_RESTRICTION_PATTERN.search(job_text)
    if not employee_match:
        return None
    return (
        f'Possible work-authorization restriction (full-time employee role): '
        f'"{employee_match.group(0)}" - check if you qualify.'
    )


def is_contract_type_role(job_text):
    """True if `job_text` explicitly reads as contract/freelance/part-time work."""
    return bool(CONTRACT_TYPE_PATTERN.search(job_text))


def is_full_time_employee_role(job_text):
    """
    Gate 5 - True if `job_text` reads as a permanent full-time EMPLOYEE
    role. An explicit contract/freelance/part-time signal always wins
    over an ambiguous "full-time" mention (e.g. "full-time contract" is
    still a contract, not employment) - only when NO contract signal is
    present does a "full-time" mention count as employment.
    """
    if is_contract_type_role(job_text):
        return False
    return bool(FULL_TIME_PATTERN.search(job_text))


def is_dead_posting(job_text):
    """Gate 4 - see DEAD_POSTING_PATTERN above."""
    return bool(DEAD_POSTING_PATTERN.search(job_text))


def is_eligibility_excluded(job_text):
    """
    Gate 5 (HARD EXCLUSION) - True if this posting should be excluded
    entirely: work_eligibility.full_time_employee_ok is false AND this
    specific posting reads as a full-time EMPLOYEE role. Used by
    update_persistent_job_list() to drop the job from the list outright
    (see the module note on Gate 5 above for why this is an exclusion,
    not a score penalty or a flag). Always False for anyone who hasn't
    set full_time_employee_ok to false in config.json.
    """
    return not WORK_ELIGIBILITY["full_time_employee_ok"] and is_full_time_employee_role(job_text)


def apply_score_adjustments(score, domain_fit, title, job_text):
    """
    Apply the role-type HARD cap, then the seniority SOFT penalty, and
    return the result (see the gate comments above for why they're
    treated differently). Location is intentionally NOT applied here at
    all - see location_flag(), used separately to add an informational
    flag without touching the score. Full-time-employee eligibility is
    ALSO not applied here - see is_eligibility_excluded() and Gate 5's
    comment above for why that's a hard exclusion elsewhere, not a score
    adjustment.
    """
    if not domain_fit or role_type_mismatch(title) or content_suggests_non_engineering_role(title, job_text):
        return min(score, DOMAIN_MISMATCH_SCORE_CAP)

    if seniority_mismatch(title, job_text):
        score = max(0, score - SENIORITY_PENALTY_POINTS)

    return score

# This is the standard local address Ollama listens on. "11434" is just
# the fixed port number Ollama always uses by default.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Your profile lives in CONFIG_FILE, NOT hardcoded here - see
# load_config()/load_profile_config() below. This keeps personal data
# (real name, employer, background) out of the codebase, so this repo
# can be shared/forked without leaking anyone's resume - see
# config.example.json for the expected shape.
CONFIG_FILE = "config.json"
CONFIG_EXAMPLE_FILE = "config.example.json"


def load_config():
    """
    Read and parse CONFIG_FILE once - both load_profile_config() and
    resolve_llm_provider() below derive their settings from this same
    dict, so config.json is only ever opened a single time per run.

    Exits with a clear, actionable message (never a raw traceback) if
    CONFIG_FILE is missing or isn't valid JSON. The fix is always the
    same: copy CONFIG_EXAMPLE_FILE to CONFIG_FILE and fill in your own
    background.
    """
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {CONFIG_FILE} not found.")
        print(f"Copy {CONFIG_EXAMPLE_FILE} to {CONFIG_FILE} and fill in your own profile:")
        print(f"  cp {CONFIG_EXAMPLE_FILE} {CONFIG_FILE}")
        sys.exit(1)
    except json.JSONDecodeError as error:
        print(f"ERROR: {CONFIG_FILE} is not valid JSON: {error}")
        sys.exit(1)


def load_profile_config(config):
    """
    Pull ("my_profile", "my_projects_by_domain") out of the parsed
    config dict (see load_config()). Exits with a clear message if
    either required key is missing - there's no sane default profile to
    fall back to, and every scoring/proposal call needs one.
    """
    try:
        return config["my_profile"], config["my_projects_by_domain"]
    except KeyError as error:
        print(f"ERROR: {CONFIG_FILE} is missing required key {error}.")
        print(f"See {CONFIG_EXAMPLE_FILE} for the expected shape.")
        sys.exit(1)


# ============================================================
# LLM provider selection - config.json's OPTIONAL "llm_provider"/
# "llm_model" fields let scoring run against a cloud LLM (Anthropic
# Claude or OpenAI GPT) instead of the free local Ollama default, for
# more accurate scoring at a small per-run cost. See ask_llm() below for
# the actual dispatch - score_job()/draft_proposal() call ONLY ask_llm(),
# never a provider-specific function directly, so the scoring logic,
# prompts, gates, and output format are 100% identical regardless of
# which provider answers the prompt.
# ============================================================

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o"

# If a cloud call (Anthropic/OpenAI) errors for any reason (rate limit,
# network blip, API outage), ask_llm() falls back to this LOCAL Ollama
# model for THAT ONE request rather than failing the whole run - a
# request-level safety net, distinct from DEFAULT_OLLAMA_MODEL (which is
# used when Ollama itself is the configured PRIMARY provider). Requires
# `ollama pull llama3.2` once, same as any other local model.
FALLBACK_OLLAMA_MODEL = "llama3.2"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Cloud replies need an explicit token cap (Ollama's /api/generate has no
# such requirement). Our replies are a compact structured block (scoring)
# or a 120-160 word proposal - 1500 tokens is comfortable headroom for
# either without being an invitation for a runaway/expensive reply.
CLOUD_LLM_MAX_TOKENS = 1500


def _default_model_for(provider):
    return {
        "ollama": DEFAULT_OLLAMA_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        "openai": DEFAULT_OPENAI_MODEL,
    }.get(provider, DEFAULT_OLLAMA_MODEL)


def resolve_llm_provider(config):
    """
    Decide which provider/model this run actually uses, applying the
    "never crash, always fall back to the free local default" rule: if
    config.json asks for a cloud provider but the matching API key
    environment variable (ANTHROPIC_API_KEY / OPENAI_API_KEY) isn't set,
    or "llm_provider" is some unrecognized value, print a clear warning
    and use Ollama instead rather than failing the whole run. API keys
    are read ONLY from the environment, never from config.json or
    anywhere else in the codebase - so a key is never at risk of being
    committed.
    """
    provider = config.get("llm_provider", DEFAULT_LLM_PROVIDER)
    model = config.get("llm_model") or _default_model_for(provider)

    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            'WARNING: config.json sets llm_provider to "anthropic" but the '
            "ANTHROPIC_API_KEY environment variable is not set. Falling back "
            "to the local Ollama model instead."
        )
        return DEFAULT_LLM_PROVIDER, DEFAULT_OLLAMA_MODEL

    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print(
            'WARNING: config.json sets llm_provider to "openai" but the '
            "OPENAI_API_KEY environment variable is not set. Falling back "
            "to the local Ollama model instead."
        )
        return DEFAULT_LLM_PROVIDER, DEFAULT_OLLAMA_MODEL

    if provider not in ("ollama", "anthropic", "openai"):
        print(
            f'WARNING: config.json\'s llm_provider "{provider}" is not '
            'recognized (expected "ollama", "anthropic", or "openai") - '
            "falling back to Ollama."
        )
        return DEFAULT_LLM_PROVIDER, DEFAULT_OLLAMA_MODEL

    return provider, model


# Defaults to "no restriction" (full_time_employee_ok=True) so anyone
# whose config.json doesn't have a "work_eligibility" section at all
# (older config files, or a user with no special work-authorization
# constraints) sees the job-type gate behave as a complete no-op - this
# is a personal constraint, not a universal rule.
DEFAULT_WORK_ELIGIBILITY = {
    "based_in": "",
    "full_time_employee_ok": True,
    "notes": "",
}


def load_work_eligibility_config(config):
    """
    Read the OPTIONAL "work_eligibility" section from the parsed config
    dict (see load_config()) and merge it over DEFAULT_WORK_ELIGIBILITY,
    so a config.json that only sets SOME of these keys still gets sane
    defaults for the rest.
    """
    return {**DEFAULT_WORK_ELIGIBILITY, **config.get("work_eligibility", {})}


_config = load_config()

# My profile and my concrete projects grouped by domain, as plain text -
# loaded from CONFIG_FILE (see load_profile_config() above). MY_PROFILE
# gets pasted straight into the scoring prompt so the model has context
# on your background; MY_PROJECTS_BY_DOMAIN is used by draft_proposal()
# to make sure it only pulls in projects that actually match the job's
# domain (e.g. never mention an LLM project in a computer-vision
# proposal, or vice versa).
MY_PROFILE, MY_PROJECTS_BY_DOMAIN = load_profile_config(_config)

# Which LLM backend/model this run actually uses (see resolve_llm_provider()
# above) - MODEL_NAME is used both by ask_llm()'s dispatch and in terminal
# messages ("Scoring each with {MODEL_NAME}...") regardless of provider.
LLM_PROVIDER, MODEL_NAME = resolve_llm_provider(_config)

# Your real-world work eligibility (see load_work_eligibility_config()
# above) - used by the job-type gate (Gate 5: is_full_time_employee_role(),
# is_eligibility_excluded(), update_persistent_job_list()) to decide
# whether full-time employee postings should be excluded entirely.
WORK_ELIGIBILITY = load_work_eligibility_config(_config)

# Scoring rules for the model to follow. This has gone through several
# rounds of fixes:
# - v1 just asked for "a match score" with no guidance on how strict to
#   be, and the model came back too generous (82/100 on a job that was
#   mostly heavy backend work I can't do alone).
# - v2 added strict weighting of mandatory/core requirements, but
#   OVERCORRECTED: the model started penalizing gaps (e.g. "no
#   backend/infra experience") even on jobs that never asked for
#   backend/infra at all - it was judging me against my whole profile's
#   weaknesses instead of against what THIS job actually needs.
# - v3 explicitly forced the model to first extract what the job
#   actually requires, and only count a gap against me if it maps to one
#   of those requirements. A skill I lack that the job never asked for
#   must NOT lower the score.
# - v4 fixes a different failure mode: v3 still scored large,
#   multi-skill jobs too generously in the middle (a 60/100 for a giant
#   desktop app - 3D avatars, voice, databases, websockets, installer -
#   where I could only realistically deliver one small slice). v3 only
#   checked "is this skill missing?", not "how much of the WHOLE job can
#   I actually cover?". v4 adds an explicit components-and-fraction step:
#   break the job into its main components, judge each one, and score
#   based on what FRACTION of the overall job I could deliver - so a
#   perfect match on 1 of 5 unrelated components stays LOW, not "medium".
#   It also adds a separate FLAGS step for red flags unrelated to skill
#   match (e.g. a budget far too small for the described scope).
# - v5 (this version) fixes a failure mode the automatic job-board
#   sources (sources.py) exposed that v4 never hit with hand-picked
#   Upwork postings: business/ops/sales roles (Customer Success Manager,
#   Executive Assistant, Business Development Rep) that merely mention
#   "AI" once in passing scored 75-83/100. The COMPONENTS breakdown was
#   the problem - the model would judge components like "communication"
#   or "process improvement" as PARTIAL matches against a research/ML
#   background, inflating the fraction for a job that was never an ML
#   role to begin with. v5 adds an explicit DOMAIN FIT gate BEFORE the
#   components breakdown: if the job's PRIMARY function isn't actually
#   ML/AI/data-science engineering, the score is capped low in code
#   (see DOMAIN_MISMATCH_SCORE_CAP), regardless of what the components
#   step would otherwise compute - the same "don't trust the model's own
#   arithmetic, verify in code" principle as compute_score_from_components().
STRICT_SCORING_GUIDELINES = """
Score as a STRICT freelance-fit judge, not an encouraging recruiter. Be
skeptical: most real jobs need several different skills, and partially
matching is not the same as being able to deliver the job. At the same
time, be FAIR: only judge the candidate against what THIS job actually
asks for, never against unrelated skills the candidate happens to lack.

Follow these steps before assigning a score:
0. FIRST, decide whether this job's PRIMARY function is actual hands-on
   machine learning / AI / data-science ENGINEERING work - building,
   training, fine-tuning, or deploying models; designing LLM/RAG
   systems; computer vision; data science modeling; or similarly
   technical ML/AI work. This is NOT the same as a job that just mentions
   "AI" as a buzzword, requires "AI fluency" as a soft skill, or belongs
   to a company whose PRODUCT is AI-powered while the ROLE itself is
   sales, customer success, business development, executive/admin
   support, accounting, marketing, recruiting, or another non-engineering
   function. If the job's primary function is NOT ML/AI/data-science
   engineering, write DOMAIN FIT: no and STOP giving it credit for
   superficial overlaps in the steps below - the final score MUST stay
   low no matter how the components step comes out (this is enforced in
   code too, not just by this instruction). Only a job whose primary
   function genuinely IS ML/AI/data-science engineering gets DOMAIN FIT:
   yes and a normal components-based score.
1. Extract a short list of what THIS job actually requires, based only
   on the job posting text. Split that list into MANDATORY/CORE
   requirements (explicitly required/mandatory, or clearly central to
   what the job is about) versus NICE-TO-HAVE requirements (secondary
   features, bonus skills, or minor details).
2. Break the job down into its main COMPONENTS - the 2-6 top-level,
   NON-OVERLAPPING skill-areas the work actually consists of (e.g. for a
   job building a desktop app with 3D avatars, voice, a database, and an
   installer, the components might be: "3D rendering/avatars",
   "voice/audio", "database design", "real-time networking",
   "installer/packaging"). Keep this to a short, high-level list (aim
   for 3-6, never more than 8), not a granular checklist. Do NOT list
   the same underlying skill twice under different names (e.g. don't
   list both "mobile app development" and "iOS and Android development"
   as separate components - that's one component, not two). Do NOT
   create a separate component for administrative/deliverable items like
   "documentation", "installation guide", or "configuration files" -
   these are just paperwork attached to the OTHER technical components,
   not a skill area of their own, and must not be counted as if they
   were.
3. For EACH component, judge whether the candidate profile can deliver
   it - BE STRICT AND LITERAL:
   - YES: that exact skill, tool, or a very close synonym of it is
     explicitly named in the candidate profile.
   - PARTIAL: the profile shows clearly transferable experience (e.g. a
     different but closely related ML framework or task), even though
     it isn't an exact match.
   - NO: the profile does not mention it and it is not closely related
     to anything the profile does mention.
   Do NOT mark a component YES or PARTIAL just because it "seems like
   something a competent engineer could figure out" or because it
   sounds generic (common examples that must be marked NO unless
   actually present in the profile: databases like PostgreSQL, web/app
   frameworks, mobile development, DevOps/installers, audio/voice
   pipelines, 3D/graphics work). If it is not in the profile, it is NO -
   being a capable engineer in general is not the same as having the
   specific skill this job needs.
   NEVER assume, infer, invent, or give credit for a skill, tool, or
   amount of experience that is not EXPLICITLY written in the candidate
   profile text below. If you are not sure whether the profile actually
   says it, treat it as NOT present - "the candidate is probably familiar
   with X" or "this is likely transferable" is not a valid basis for YES
   or PARTIAL unless the profile text itself supports it. Judge only the
   concrete overlap between what the profile actually lists and what the
   job actually states.
4. Count the components: FRACTION = (number of YES + 0.5 x number of
   PARTIAL) / (total number of components). Compute this as an actual
   number, do not guess or default to "about half" - a job where the
   candidate is YES on 1 of 5 components and NO on the rest has a
   fraction of 0.2, not 0.5.
5. Compare the candidate profile against the MANDATORY/CORE requirements
   from step 1 too. A "gap" only counts if it maps to something on that
   list - a skill the job never mentioned or implied is IRRELEVANT and
   must NOT be listed as a gap or lower the score.
6. Classify EACH gap you list as either:
   - (quick to learn): a SPECIFIC, narrow tool/library/framework/API/
     platform a competent engineer with the candidate's existing
     background could realistically pick up in days to a few weeks
     (e.g. "Docker", "a specific cloud provider's SDK", "GraphQL", "a
     particular JS framework", "Terraform basics").
   - (large gap): something that takes YEARS to build, not weeks - a
     fundamentally different domain the candidate has no foothold in,
     a specific number of years of professional/production experience,
     a deep specialization, or a body of skills rather than one tool
     (e.g. "5+ years of production MLOps experience", "deep expertise
     in distributed systems", "a background in embedded/robotics").
   When genuinely unsure whether a gap is quick or large, default to
   (large gap) - it's the safer, less overclaiming label.

Use these score anchors, based on the FRACTION from step 4 (this
overrides any temptation to give a "medium" score just because SOME
skill matched):
- 80-100: the candidate could deliver ALL or nearly all components
  (roughly 90%+) mostly alone, including mandatory/core requirements.
- 60-79: the candidate could deliver MOST components (roughly 60-90%)
  but is missing some real ground.
- 40-59: the candidate could deliver ABOUT HALF the components (roughly
  40-60%) - a genuine partial fit, not just "has one relevant skill".
- 20-39: the candidate could only deliver a SMALL SLICE of the overall
  job (a minority of components, e.g. 1 out of 4-5, even if that one
  slice is a strong, direct skill match) - most of the job is still
  outside their ability to deliver alone.
- 0-19: the candidate could deliver almost none of the job's components.

Separately from the score, check for FLAGS - warning signs about the JOB
ITSELF, independent of whether the candidate's skills match:
- The stated budget/pay is clearly too small for the described scope
  (e.g. a fixed price of $10-50 for a multi-month, multi-skill build).
- The scope described is large enough that it would normally need a
  small team or multiple specialists, not one freelancer.
- Any other clear mismatch between what's asked for and what a single
  freelancer could reasonably deliver in the stated timeframe/budget.
If none of these apply, there are no flags - do not invent one just to
have something to say.
"""

# The exact format we want the model to reply in. Spelling this out in
# the prompt (instead of just asking a free-form question) makes the
# model's reply predictable and easy to read - and easier to parse with
# code later, if we want to extract just the score, for example.
RESPONSE_FORMAT_INSTRUCTIONS = """
Respond in EXACTLY this format, with nothing before or after it, and no
markdown formatting (no asterisks, no headers). Fill in DOMAIN FIT,
JOB REQUIRES, and COMPONENTS FIRST, and only list a gap in GAPS if it
also appears in JOB REQUIRES - if a candidate weakness is not in JOB
REQUIRES, it must be left out of GAPS entirely, since the job never
asked for it:

DOMAIN FIT: <yes/no - is this job's PRIMARY function genuine ML/AI/data-science engineering work, not a business/ops/sales/support role that just mentions AI in passing?>
JOB REQUIRES: <comma-separated list of what this job actually needs, mandatory items marked with (core)>
COMPONENTS: <comma-separated list of the job's main, non-overlapping components, each followed by (yes), (partial), or (no) - be strict, see the rules above>
DELIVERABLE FRACTION: <the actual count, e.g. "1 YES + 0 PARTIAL out of 5 components = 0.2">
SCORE: <a single integer from 0 to 100, computed from the fraction above (fraction x 100, adjusted slightly for mandatory/core gaps) - if DOMAIN FIT is no, this must be 15 or below regardless of the fraction>
STRENGTHS:
- <strength 1>
- <strength 2>
- <strength 3 (optional)>
GAPS:
- <gap - must be an item from JOB REQUIRES that the candidate lacks> (quick to learn|large gap)
- <gap 2 (optional, same rule)> (quick to learn|large gap)
FLAGS:
- <a warning sign about the job itself, e.g. budget too low for the scope (optional - write "None" if there are no flags)>
VERDICT: <one short sentence>
"""


def ask_ollama(prompt, temperature=0, model=None):
    """
    Send `prompt` to the local Ollama server and return the model's
    text reply as a string. `model` overrides MODEL_NAME for this one
    call - used by ask_llm()'s cloud-provider fallback path to force
    FALLBACK_OLLAMA_MODEL regardless of what the primary provider is.

    How it works:
    - Ollama's /api/generate endpoint expects a JSON body with at least
      "model" (which model to use) and "prompt" (what to ask it).
    - By default, Ollama STREAMS its reply back piece by piece. To keep
      this first test as simple as possible, we set "stream": False so
      Ollama waits and sends back the whole answer in one single
      response instead of many small chunks.
    - "temperature" controls how random the model's word choices are.
      Higher temperature (e.g. 0.7-1.0) makes replies more varied and
      "creative" - good for brainstorming, bad for scoring, because the
      same job could get a different score every time we ask. Setting
      temperature to 0 makes the model always pick its single most
      likely next word, which is what we want for a scoring task: the
      same (profile, job) pair should produce the same score every run,
      so the ranking is trustworthy and comparable across jobs and over
      time. We pass it inside an "options" object, which is where
      Ollama expects generation settings like this to live.
    - requests.post(...) sends the HTTP request; .json() parses the
      JSON reply into a normal Python dictionary.
    - The actual generated text lives under the "response" key of that
      dictionary, so that's what we return.
    """
    request_body = {
        "model": model or MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(OLLAMA_URL, json=request_body)

    # Raise an error early if Ollama responded with a failure status
    # code (e.g. model not found, server not running), instead of
    # silently continuing with a broken response.
    response.raise_for_status()

    response_data = response.json()
    return response_data["response"]


def ask_anthropic(prompt, temperature=0):
    """
    Send `prompt` to the Anthropic Messages API (Claude) and return the
    reply text. Only ever called when LLM_PROVIDER is "anthropic", which
    resolve_llm_provider() only sets after confirming ANTHROPIC_API_KEY
    is present - so this doesn't need to re-check for the key itself.

    Uses `requests` directly (the one dependency this project already
    has) rather than adding the anthropic SDK as a second dependency,
    same reasoning as the raw HTTP call to Ollama above.
    """
    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "max_tokens": CLOUD_LLM_MAX_TOKENS,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def ask_openai(prompt, temperature=0):
    """
    Send `prompt` to the OpenAI Chat Completions API (GPT) and return
    the reply text. Only ever called when LLM_PROVIDER is "openai",
    which resolve_llm_provider() only sets after confirming
    OPENAI_API_KEY is present.
    """
    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_llm(prompt, temperature=0):
    """
    Send `prompt` to whichever LLM provider config.json selects (see
    resolve_llm_provider()) and return (reply_text, model_label). This
    is the ONLY function score_job()/draft_proposal() call - swapping
    providers means changing config.json, never touching scoring logic,
    prompts, gates, or output format.

    `model_label` records EXACTLY which backend/model produced THIS
    reply (e.g. "anthropic:claude-haiku-4-5-20251001") - see
    run_scoring_and_report()'s "model" cache field. This can differ from
    what config.json asks for: if a cloud call errors (rate limit,
    network blip, API outage), this call falls back to the local
    FALLBACK_OLLAMA_MODEL for just this one request instead of failing
    the whole run, and the label reflects that ("ollama:llama3.2
    (fallback)") so the cache never lies about which model actually
    answered.
    """
    if LLM_PROVIDER == "anthropic":
        try:
            return ask_anthropic(prompt, temperature=temperature), f"anthropic:{MODEL_NAME}"
        except requests.exceptions.RequestException as error:
            print(
                f"WARNING: Anthropic API call failed ({error}) - falling back to "
                f"local Ollama ({FALLBACK_OLLAMA_MODEL}) for this request."
            )
            reply = ask_ollama(prompt, temperature=temperature, model=FALLBACK_OLLAMA_MODEL)
            return reply, f"ollama:{FALLBACK_OLLAMA_MODEL} (fallback)"

    if LLM_PROVIDER == "openai":
        try:
            return ask_openai(prompt, temperature=temperature), f"openai:{MODEL_NAME}"
        except requests.exceptions.RequestException as error:
            print(
                f"WARNING: OpenAI API call failed ({error}) - falling back to "
                f"local Ollama ({FALLBACK_OLLAMA_MODEL}) for this request."
            )
            reply = ask_ollama(prompt, temperature=temperature, model=FALLBACK_OLLAMA_MODEL)
            return reply, f"ollama:{FALLBACK_OLLAMA_MODEL} (fallback)"

    return ask_ollama(prompt, temperature=temperature), f"ollama:{MODEL_NAME}"


def score_job(job_text, temperature=0):
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

    Returns (raw_reply, model_label) - raw_reply is the model's reply
    string (already in the format described above), model_label records
    which backend/model actually produced it (see ask_llm()).

    `temperature` defaults to 0 for the normal, deterministic case (see
    the comment on ask_ollama() for why). run_scoring_and_report() passes
    a small non-zero temperature for a one-time RETRY when the model's
    first reply didn't include the (yes)/(partial)/(no) COMPONENTS
    labels compute_score_from_components() needs - at temperature=0,
    retrying with the exact same prompt would just reproduce the same
    malformed reply, so the retry needs a little randomness to have any
    chance of getting a differently-shaped (hopefully compliant) answer.
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

    return ask_llm(prompt, temperature=temperature)


def draft_proposal(job_text):
    """
    Ask the model to write a short Upwork proposal for `job_text`, based
    on MY_PROFILE.

    Unlike score_job(), this call uses a HIGHER temperature (0.7) on
    purpose. Scoring needs to be consistent every time (temperature=0),
    but writing needs some natural variation in wording and phrasing to
    avoid sounding stiff and robotic - a bit of randomness here is a
    feature, not a bug.

    The earlier version of this prompt just said "pick whichever skills
    are most relevant" without forcing the model to commit to a domain
    first - so on a computer-vision job, it sometimes still mentioned
    LLM fine-tuning, because that's an impressive-sounding item
    elsewhere in my profile, even though it has nothing to do with the
    job. This version fixes that by making domain classification an
    explicit first step, and restricting the proposal to only that
    domain's skills/projects (from MY_PROJECTS_BY_DOMAIN).

    The prompt asks for a short (120-160 word) proposal that:
    - first identifies the job's domain (e.g. computer vision vs.
      LLM/NLP vs. backend/infra), silently, before writing anything
    - opens with something specific to this job, not a generic greeting
    - names 2-3 of my most relevant skills/projects for THIS job's
      domain only - never an impressive but unrelated one
    - stays honest - never claims a skill that isn't in MY_PROFILE
    - ends with a simple call to action
    - reads like a warm, human message, not an overly formal cover letter

    Returns the drafted proposal as a plain string.
    """
    prompt = f"""You are me, a freelancer, writing a short Upwork proposal to apply
to the job posting below. Write in first person, as if I'm writing it
myself.

CANDIDATE PROFILE (this is who I am - only mention skills/experience
that actually appear here; never invent or exaggerate anything):
{MY_PROFILE}

MY PROJECTS, GROUPED BY DOMAIN:
{MY_PROJECTS_BY_DOMAIN}

JOB POSTING:
{job_text}

Before writing anything, decide which ONE domain this job is mainly
about (for example: computer vision, LLM/NLP, or backend/infra). Then
write the proposal using ONLY skills and projects from that matching
domain above. Do not mention skills or projects from a different,
unrelated domain, even if they sound impressive - e.g. never mention
LLM/NLP work (GPT from scratch, LoRA, RAG, Hugging Face) in a
computer-vision proposal, and never mention computer-vision work
(ResNet, CNNs, EuroSAT, plant classification) in an LLM/NLP proposal.

Write a proposal that:
- Is 120-160 words long.
- Opens by addressing the client's specific need described in the job
  posting - not a generic "I am excited to apply" opener.
- Names 2-3 of my most relevant skills or projects FOR THIS SPECIFIC
  JOB'S DOMAIN ONLY (see the domain rule above).
- Is completely honest - do NOT claim any skill, tool, or experience
  that isn't in my profile or projects above.
- Ends with a simple, low-pressure call to action (e.g. inviting a
  quick chat or asking a clarifying question).
- Sounds human and warm, like a real person wrote it - not robotic,
  not overly formal, no corporate buzzwords.

Respond with ONLY the proposal text itself - no preamble, no headers,
no notes about word count or which domain you picked.
"""

    # temperature=0.7 (higher than scoring) so the writing sounds natural
    # instead of terse/robotic - see the docstring above for why. The
    # model label isn't tracked for proposals (only the scoring cache
    # needs "model" - see PART 1/run_scoring_and_report()).
    raw_reply, _model_label = ask_llm(prompt, temperature=0.7)

    # The prompt above already asks for no preamble, but an 8B model
    # doesn't always obey that - strip_proposal_preamble() is the
    # reliable guarantee, on top of (not instead of) the prompt instruction.
    return strip_proposal_preamble(raw_reply)


# A small set of phrases that signal the model is narrating ABOUT the
# proposal ("Here is the proposal:", "Sure, I'd be happy to help...")
# rather than writing the proposal itself. Only used to recognize
# leading paragraphs to remove - see strip_proposal_preamble() below.
PREAMBLE_PATTERNS = [
    r"^here'?s?\s+(is\s+)?(the|your|my)?\s*proposal",
    r"^here'?s my attempt",
    r"^i'?d be happy to help",
    r"^sure[,!]",
    r"^okay,?\s",
    r"^based on the job posting",
    r"^let me (write|draft)",
]


def strip_proposal_preamble(raw_text):
    """
    Remove any leading narration the model added before the actual
    proposal (e.g. "Here is the proposal:" or "I'd be happy to help...
    Here's my attempt..."), as a reliable backstop for when the model
    ignores the "no preamble" instruction in the prompt.

    How it works:
    - Split the reply into paragraphs, wherever there's a blank line.
    - Look at paragraphs from the top. If a paragraph's text matches one
      of the PREAMBLE_PATTERNS above (the model talking ABOUT the
      proposal, not writing it), drop that paragraph and check the next
      one the same way.
    - Stop as soon as we reach a paragraph that does NOT match a
      preamble pattern - that's where the real proposal starts, and we
      leave everything from there onward untouched.

    This only ever removes paragraphs from the very top of the reply,
    and only ones that match a known lead-in phrase, so real proposal
    content (which won't match these patterns) is never at risk of
    being deleted.
    """
    paragraphs = raw_text.strip().split("\n\n")

    while len(paragraphs) > 1:
        first_paragraph = paragraphs[0].strip()
        looks_like_preamble = any(
            re.match(pattern, first_paragraph, re.IGNORECASE) for pattern in PREAMBLE_PATTERNS
        )
        if looks_like_preamble:
            paragraphs.pop(0)
        else:
            break

    return "\n\n".join(paragraphs).strip()


def load_jobs(file_path):
    """
    Read `file_path` (jobs.json, written by server.py) and return its
    contents as a list of job dicts, each with at least "text" and
    "date_added" keys.

    If the file doesn't exist yet (no jobs saved so far) or somehow
    isn't valid JSON, we return an empty list instead of crashing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    return jobs


def extract_title_and_metadata(job_text):
    """
    Pull a short title and (if present) a metadata line out of a job's
    raw text, for display purposes only (this doesn't affect scoring).

    - Title: always just the job's first line, so both old plain-text
      jobs (a single unbroken paragraph) and jobs captured by the
      browser extension (title on its own first line) work the same way.
    - Metadata: the browser extension writes job text as
      "title\\n\\nExperience level: ... | Pricing type: ... \\n\\ndescription...".
      If the SECOND paragraph (split on a blank line) looks like that
      metadata line - mentioning things like "Experience level" or
      "Hours per week" - we treat it as metadata. Older jobs that were
      typed/pasted by hand won't have this paragraph, so metadata simply
      comes back as None for them, which is fine.
    """
    stripped_text = job_text.strip()
    title = stripped_text.splitlines()[0].strip() if stripped_text else ""

    metadata = None
    paragraphs = stripped_text.split("\n\n")
    if len(paragraphs) > 1:
        second_paragraph = paragraphs[1].strip()
        if re.search(r"Experience level|Pricing type|Hours per week|Duration", second_paragraph):
            metadata = second_paragraph

    return title, metadata


def save_jobs(jobs):
    """
    Write `jobs` (a list of job dicts) back to JOBS_JSON_FILE. Used by
    --delete to persist a job's removal. We always rewrite the whole
    file, since JSON is one big list rather than something we can
    append/remove single lines from.
    """
    with open(JOBS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def hash_job_text(job_text):
    """
    Turn a job's text into a short, stable identifier (a SHA-256 hash),
    used as its key in results.json. The same job text always produces
    the same hash, which is exactly what we want for caching: "have I
    already scored this exact job before?"
    """
    return hashlib.sha256(job_text.strip().encode("utf-8")).hexdigest()


def load_results():
    """
    Read RESULTS_JSON_FILE and return its contents as a dict mapping
    job-text hash -> cached result (title, score, verdict, proposal).
    Returns an empty dict if the file doesn't exist yet or isn't valid
    JSON, so a missing/corrupt cache never crashes a run - it just means
    everything gets (re-)scored.
    """
    try:
        with open(RESULTS_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_job_list():
    """
    Read JOB_LIST_FILE and return its contents as a dict with "jobs"
    (hash -> entry) and "removed_hashes" (list) keys always present,
    even if the file is missing/corrupt (first-ever run) or was written
    by an older version that didn't have one of these keys yet.
    """
    try:
        with open(JOB_LIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.setdefault("jobs", {})
    data.setdefault("removed_hashes", [])
    return data


def save_job_list(job_list):
    """Write `job_list` (see load_job_list()) back to JOB_LIST_FILE."""
    with open(JOB_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(job_list, f, indent=2, ensure_ascii=False)


def dedupe_job_list_by_url(current_jobs):
    """
    Collapse any entries in `current_jobs` (JOB_LIST_FILE's "jobs" dict)
    that share the same non-empty "url" down to a single entry - the
    highest-scoring one. Mutates `current_jobs` in place and returns how
    many entries were dropped.

    Why this exists: the normalized title+company dedup key (see
    sources.normalize_company_title_key()) missed a real duplicate - the
    same Himalayas posting was fetched once with company "name" (a stale
    placeholder from earlier testing) and once with the real company
    "Featherless AI", producing two different keys AND two different
    exact-text hashes for the same job (also scored differently - 75 vs
    90 - across the two fetches, since the local 8B model isn't fully
    consistent run-to-run; the gates/prompt are unchanged, that's just
    inherent model noise, worth knowing about but not something dedup
    can fix). A job's URL is a far more reliable "is this the same
    posting" signal than its title/company text, so this is checked
    independently as a second dedup pass.
    """
    best_hash_by_url = {}
    for job_hash, entry in current_jobs.items():
        url = entry.get("url")
        if not url:
            continue
        current_best = best_hash_by_url.get(url)
        if current_best is None or entry["score"] > current_jobs[current_best]["score"]:
            best_hash_by_url[url] = job_hash

    hashes_to_drop = [
        job_hash
        for job_hash, entry in current_jobs.items()
        if entry.get("url") and best_hash_by_url[entry["url"]] != job_hash
    ]
    for job_hash in hashes_to_drop:
        del current_jobs[job_hash]

    return len(hashes_to_drop)


def remove_ineligible_jobs(current_jobs, jobs_by_hash):
    """
    Retroactively drop any EXISTING persistent-list entries that read as
    full-time EMPLOYEE roles, now that Gate 5 (see is_eligibility_excluded())
    is a hard exclusion rather than the soft penalty/flag it used to be -
    without this, a job added to the list under the old behavior would
    linger forever even though it's no longer eligible to be shown.
    Mutates `current_jobs` in place and returns how many were dropped.

    `jobs_by_hash` maps job hash -> job text (built from jobs.json by the
    caller), since job_list.json's entries don't store the full text
    themselves - only jobs.json does.
    """
    if WORK_ELIGIBILITY["full_time_employee_ok"]:
        return 0  # no restriction configured - nothing to retroactively remove

    to_remove = [
        job_hash
        for job_hash in current_jobs
        if job_hash in jobs_by_hash and is_eligibility_excluded(jobs_by_hash[job_hash])
    ]
    for job_hash in to_remove:
        del current_jobs[job_hash]

    return len(to_remove)


def update_persistent_job_list():
    """
    Add newly-qualifying automatically-sourced jobs to the persistent
    JOB_LIST_FILE and save it. Called once per --daily run, after
    scoring. Returns (job_list, added_count, deduped_count).

    A job earns a spot in the list if ALL of:
    - it came from an automatic source (has "freshness" - the manual
      Upwork-extension flow is out of scope for this list, same as the
      old cutoff-based daily_report.html was).
    - it's been scored (has a results.json entry).
    - its score is ABOVE FIT_SCORE_THRESHOLD.
    - its title doesn't trip the role-type hard-cap gate (belt and
      suspenders alongside the score check - a role-type mismatch
      already caps the score below the threshold via
      apply_score_adjustments(), so this should never independently
      change the outcome, but checking it explicitly costs nothing and
      guards against the threshold being tuned down later).
    - it's NOT excluded by the work-eligibility gate (see
      is_eligibility_excluded() - a full-time EMPLOYEE posting when
      you're only eligible for part-time/remote/freelance/contract work
      is dropped here entirely, not scored down or flagged).
    - its URL isn't already covered by a higher (or equally) scoring
      entry already in the list (see dedupe_job_list_by_url() above) -
      if it scores HIGHER than the existing entry for that same URL,
      the existing one is replaced rather than listing both.

    Once added, a job's hash stays in the list PERMANENTLY across runs -
    this function never removes anything else (that's cli_run_daily's
    caller via server.py's /remove_daily_job and /clear_daily_list) - and
    a hash already in "removed_hashes" (a job you deleted before) is
    never re-added, even if a later fetch re-discovers the same posting.
    """
    jobs = load_jobs(JOBS_JSON_FILE)
    results = load_results()
    job_list = load_job_list()

    removed_hashes = set(job_list["removed_hashes"])
    current_jobs = job_list["jobs"]
    jobs_by_hash = {hash_job_text(j.get("text", "")): j.get("text", "") for j in jobs}

    # Retroactively clean up any URL duplicates, and any now-ineligible
    # full-time-employee entries, already in the list before considering
    # new jobs (see dedupe_job_list_by_url()/remove_ineligible_jobs()).
    deduped_count = dedupe_job_list_by_url(current_jobs)
    remove_ineligible_jobs(current_jobs, jobs_by_hash)

    url_to_hash = {entry["url"]: h for h, entry in current_jobs.items() if entry.get("url")}

    added = 0
    for job in jobs:
        if "freshness" not in job:
            continue  # manual Upwork-extension flow - out of scope here

        job_text = job.get("text", "")
        job_hash = hash_job_text(job_text)
        if job_hash in current_jobs or job_hash in removed_hashes:
            continue

        cached = results.get(job_hash)
        if not cached:
            continue  # not scored yet - will be picked up on a future run

        title = cached["title"]
        if cached["score"] <= FIT_SCORE_THRESHOLD or role_type_mismatch(title):
            continue

        if is_eligibility_excluded(job_text):
            continue  # full-time employee role, not eligible - excluded entirely

        url = job.get("url", "")
        existing_hash = url_to_hash.get(url) if url else None
        if existing_hash is not None:
            if cached["score"] <= current_jobs[existing_hash]["score"]:
                continue  # same posting, already covered by an equal-or-better entry
            del current_jobs[existing_hash]  # same posting, but this one scored higher - replace it

        current_jobs[job_hash] = {
            "title": title,
            "score": cached["score"],
            "verdict": cached["verdict"],
            "flags": cached.get("flags", []),
            "gaps": cached.get("gaps", []),
            "source": job.get("source", "upwork-extension"),
            "url": url,
            "date_posted": job.get("date_added", ""),
            "date_found": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if url:
            url_to_hash[url] = job_hash
        added += 1

    save_job_list(job_list)
    return job_list, added, deduped_count


def save_results(results):
    """
    Write the `results` cache dict back to RESULTS_JSON_FILE.
    """
    with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_score_and_verdict(raw_reply):
    """
    Pull just the SCORE and VERDICT values out of a score_job() reply,
    so we can rank jobs without re-printing the model's full response
    for each one.

    We use simple regexes to find "SCORE: <number>" and
    "VERDICT: <text>" anywhere in the reply. If the model didn't follow
    the format for some reason and a piece can't be found, we fall back
    to a 0 score / "(no verdict found)" placeholder instead of crashing,
    so one malformed reply doesn't stop the whole ranking.
    """
    score_match = re.search(r"SCORE:\s*(\d+)", raw_reply)
    verdict_match = re.search(r"VERDICT:\s*(.+)", raw_reply)

    score = int(score_match.group(1)) if score_match else 0
    verdict = verdict_match.group(1).strip() if verdict_match else "(no verdict found)"

    return score, verdict


def parse_domain_fit(raw_reply):
    """
    Pull the "DOMAIN FIT: yes/no" line out of a score_job() reply (see
    STRICT_SCORING_GUIDELINES' v5 note and DOMAIN_MISMATCH_SCORE_CAP).

    Returns True only if the model explicitly wrote "yes" - a missing or
    malformed DOMAIN FIT line does NOT default to True, since a job we
    can't confirm is genuinely ML/AI/data-science engineering shouldn't
    get the benefit of the doubt on a score cap this consequential.
    """
    match = re.search(r"DOMAIN FIT:\s*(yes|no)", raw_reply, re.IGNORECASE)
    return bool(match) and match.group(1).lower() == "yes"


def parse_flags(raw_reply):
    """
    Pull the bullet lines under "FLAGS:" out of a score_job() reply -
    warning signs about the JOB ITSELF (e.g. a budget far too small for
    the described scope), independent of the skill-fit score.

    Returns a list of flag strings, or an empty list if the model wrote
    "None" (no flags for this job) or the FLAGS section wasn't found at
    all - either way, no flags is a perfectly normal, common result.
    """
    # Capture everything between "FLAGS:" and the next "VERDICT:" line
    # (or the end of the text, if VERDICT wasn't found for some reason).
    flags_match = re.search(r"FLAGS:\s*(.*?)(?:\n\s*VERDICT:|\Z)", raw_reply, re.DOTALL)
    if not flags_match:
        return []

    flags = []
    for line in flags_match.group(1).splitlines():
        flag_text = line.strip().lstrip("-").strip()
        if flag_text and flag_text.lower() not in ("none", "none.", "n/a"):
            flags.append(flag_text)

    return flags


GAP_CATEGORY_PATTERN = re.compile(r"^(.*?)\s*\((quick to learn|large gap)\)", re.IGNORECASE)


def parse_gaps(raw_reply):
    """
    Pull the bullet lines under "GAPS:" out of a score_job() reply - see
    RESPONSE_FORMAT_INSTRUCTIONS/STRICT_SCORING_GUIDELINES step 6 for the
    "(quick to learn)"/"(large gap)" label every gap is supposed to
    carry. Used by the skill-gap-analysis feature (see
    compute_skill_gap_summary() and build_daily_job_card_html()) to tell
    a genuine near-miss (missing one learnable tool) from a job that was
    never realistically in reach.

    Returns a list of {"text": <gap description>, "category": "quick" |
    "large"} dicts. A gap line the model forgot to label (or the model's
    label didn't match either exact phrase) defaults to "large" - the
    same "when unsure, don't overclaim it's easy" rule the prompt itself
    is given, applied again here as a code-level backstop.
    """
    gaps_match = re.search(r"GAPS:\s*(.*?)(?:\n\s*FLAGS:|\Z)", raw_reply, re.DOTALL)
    if not gaps_match:
        return []

    gaps = []
    for line in gaps_match.group(1).splitlines():
        line_text = line.strip().lstrip("-").strip()
        if not line_text or line_text.lower() in ("none", "none.", "n/a"):
            continue

        category_match = GAP_CATEGORY_PATTERN.match(line_text)
        if category_match:
            gap_text = category_match.group(1).strip()
            category = "quick" if "quick" in category_match.group(2).lower() else "large"
        else:
            gap_text = line_text
            category = "large"

        if gap_text:
            gaps.append({"text": gap_text, "category": category})

    return gaps


# Component names matching any of these are administrative/deliverable
# items (documentation, installation guides, config files, etc.), not
# real skill areas. The prompt already tells the model not to list these
# as their own components, but an 8B model doesn't always obey that -
# this is a code-level backstop that filters them out regardless of what
# the model did, so a job with a long "write 5 kinds of documentation"
# list can't inflate its component count with easy "yes" items.
BOILERPLATE_COMPONENT_PATTERN = re.compile(
    r"documentation|installation guide|install guide|setup guide|user guide"
    r"|instructions|installer|readme|config(uration)? files?",
    re.IGNORECASE,
)


def compute_score_from_components(raw_reply):
    """
    Compute the match score directly from the model's COMPONENTS
    classification (see RESPONSE_FORMAT_INSTRUCTIONS), instead of
    trusting the SCORE number the model wrote itself.

    Why: an 8B model can classify a component as (yes)/(partial)/(no)
    reasonably well, but is unreliable at then doing the arithmetic to
    turn that into a percentage - in testing it sometimes wrote a
    DELIVERABLE FRACTION calculation with the wrong answer (e.g. "1 YES
    + 0.5x2 PARTIAL out of 12 = 0.583" when that arithmetic is actually
    0.167), and its SCORE line inherited that wrong number. Since the
    (yes)/(partial)/(no) labels themselves are just text we can count
    reliably with code, we do the arithmetic ourselves instead of asking
    the model to:

        fraction = (yes_count + 0.5 * partial_count) / total_count
        score = round(fraction * 100)

    We also drop any component whose NAME matches
    BOILERPLATE_COMPONENT_PATTERN before counting - the model sometimes
    ignores the prompt's instruction not to list documentation/installer
    items as their own components (and marks them an easy "yes"), which
    would otherwise pad the fraction with components that aren't real
    engineering work.

    Returns None if no usable (non-boilerplate) "name (yes/partial/no)"
    pairs can be found, so the caller can fall back to the model's own
    SCORE line instead of crashing or dividing by zero.
    """
    components_match = re.search(r"COMPONENTS:\s*(.+)", raw_reply)
    if not components_match:
        return None

    # Capture each "component name (yes/partial/no)" pair together, so
    # we can filter by name before counting labels.
    pairs = re.findall(r"([^,()]+?)\s*\((yes|partial|no)\)", components_match.group(1), re.IGNORECASE)
    if not pairs:
        return None

    real_labels = [
        label.lower() for name, label in pairs if not BOILERPLATE_COMPONENT_PATTERN.search(name)
    ]
    if not real_labels:
        return None

    yes_count = real_labels.count("yes")
    partial_count = real_labels.count("partial")
    total_count = len(real_labels)

    fraction = (yes_count + 0.5 * partial_count) / total_count
    return round(fraction * 100)


def score_to_color(score):
    """
    Map a 0-100 score to a hex color for the HTML report:
    green (>= 70), orange (50-69), red (< 50). A score of None (a job
    that hasn't been scored yet - see build_report()) gets a neutral gray.
    """
    if score is None:
        return "#888888"  # gray - not scored yet
    elif score >= 70:
        return "#1b8a3d"  # green
    elif score >= 50:
        return "#c77700"  # orange
    else:
        return "#c53030"  # red


def format_date_heading(date_added):
    """
    Turn a "YYYY-MM-DD" string into a friendly heading like
    "July 17, 2026". Falls back to the raw string unchanged if it's
    missing or not in the expected format, rather than crashing the
    whole report over one malformed date.
    """
    try:
        return datetime.datetime.strptime(date_added, "%Y-%m-%d").strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return date_added or "Unknown date"


def build_job_card_html(entry):
    """
    Build the HTML for a single job "card" (rank, color-coded score,
    title, metadata line if present, verdict, and drafted proposal if
    there is one).

    We use html.escape() on every piece of text that came from the job
    posting or the model's reply, since that text is unpredictable and
    could otherwise contain characters (like "<" or "&") that would
    break the HTML page's structure.

    entry["score"] may be None for a job that hasn't been scored yet
    (see build_report()) - shown as "Not scored yet" instead of a number.
    """
    color = score_to_color(entry["score"])
    score_label = "Not scored yet" if entry["score"] is None else f"{entry['score']}/100"

    metadata_html = ""
    if entry["metadata"]:
        metadata_html = f'<p class="metadata">{html.escape(entry["metadata"])}</p>'

    flags_html = ""
    if entry["flags"]:
        flag_items_html = "".join(f"<li>{html.escape(flag)}</li>" for flag in entry["flags"])
        flags_html = f"""
        <div class="flags-box">
            <div class="flags-label">⚠ Flags</div>
            <ul>{flag_items_html}</ul>
        </div>
        """

    proposal_html = ""
    if entry["proposal"]:
        # Convert newlines in the proposal text to <br> so paragraph
        # breaks show up correctly in the browser.
        proposal_text_html = html.escape(entry["proposal"]).replace("\n", "<br>")
        proposal_html = f"""
        <div class="proposal-box">
            <div class="proposal-label">Drafted proposal</div>
            <p>{proposal_text_html}</p>
        </div>
        """

    # data-job-id carries this job's text hash (see hash_job_text()), so
    # the delete button's JavaScript (in build_html_report()) can tell
    # server.py's /delete_job endpoint exactly which job to remove,
    # without needing to send the whole job text back.
    return f"""
    <div class="card" data-job-id="{entry['job_id']}">
        <div class="card-header">
            <span class="rank">#{entry['rank']}</span>
            <span class="score" style="color: {color};">{score_label}</span>
        </div>
        <h3 class="job-title">{html.escape(entry['title'])}</h3>
        {metadata_html}
        <p class="verdict">{html.escape(entry['verdict'])}</p>
        {flags_html}
        {proposal_html}
        <button class="delete-button" data-job-id="{entry['job_id']}">🗑 Delete</button>
    </div>
    """


def build_html_report(results_by_date, sorted_dates):
    """
    Build a full HTML page (as a string), grouped into one section per
    date (newest first), with each date's jobs shown as cards ranked
    high-to-low by score within that section.

    `results_by_date` is a dict: {"YYYY-MM-DD": [entry, entry, ...]},
    where each entry is a dict with keys "rank", "score", "title",
    "metadata", "verdict", "proposal" (metadata/proposal may be None).
    `sorted_dates` is the list of those date keys, already ordered
    newest-first - see the main block below for how both are built.
    """
    generated_at_str = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    sections_html = ""
    for date_added in sorted_dates:
        date_heading = format_date_heading(date_added)
        cards_html = "".join(build_job_card_html(entry) for entry in results_by_date[date_added])
        sections_html += f"""
        <section class="date-section">
            <h2 class="date-heading">{html.escape(date_heading)}</h2>
            {cards_html}
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Upwork Job Matches</title>
<style>
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
        background-color: #f4f5f7;
        color: #22262b;
        max-width: 820px;
        margin: 40px auto;
        padding: 0 20px 60px 20px;
        line-height: 1.5;
    }}
    .page-header {{
        margin-bottom: 32px;
    }}
    h1 {{
        font-size: 26px;
        margin: 0 0 6px 0;
    }}
    .generated-at {{
        font-size: 13px;
        color: #777;
        margin: 0;
    }}
    .date-section {{
        margin-bottom: 36px;
    }}
    .date-heading {{
        font-size: 18px;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #ddd;
        padding-bottom: 8px;
        margin-bottom: 18px;
    }}
    .card {{
        background-color: #ffffff;
        border: 1px solid #e2e2e2;
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 15px;
        font-weight: 600;
    }}
    .rank {{
        color: #888;
    }}
    .score {{
        font-size: 21px;
        font-weight: 700;
    }}
    .job-title {{
        font-size: 17px;
        font-weight: 600;
        margin: 10px 0 4px 0;
    }}
    .metadata {{
        font-size: 13px;
        color: #666;
        margin: 0 0 10px 0;
    }}
    .verdict {{
        margin: 0 0 10px 0;
        color: #444;
    }}
    .proposal-box {{
        background-color: #f7f8fa;
        border-left: 4px solid #888;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 14px;
    }}
    .proposal-label {{
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #666;
        margin-bottom: 6px;
    }}
    .flags-box {{
        background-color: #fff8e6;
        border-left: 4px solid #d9a441;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 10px 0;
        font-size: 13px;
        color: #6b4e14;
    }}
    .flags-label {{
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #8a6215;
        margin-bottom: 4px;
    }}
    .flags-box ul {{
        margin: 0;
        padding-left: 18px;
    }}
    .delete-button {{
        margin-top: 14px;
        padding: 6px 12px;
        font-size: 13px;
        border: 1px solid #e0b4b4;
        border-radius: 6px;
        background-color: #fdf2f2;
        color: #a33;
        cursor: pointer;
    }}
    .delete-button:hover {{
        background-color: #fbe4e4;
    }}
</style>
</head>
<body>
    <div class="page-header">
        <h1>Upwork Job Matches</h1>
        <p class="generated-at">Generated {generated_at_str}</p>
    </div>
    {sections_html}

    <script>
        // This runs in the browser when report.html is opened. It wires
        // up every "Delete" button to call server.py's /delete_job
        // endpoint (server.py must be running for this to work - see
        // the comment on DELETE_URL below).
        const DELETE_URL = "http://localhost:8765/delete_job";

        document.querySelectorAll(".delete-button").forEach(function (button) {{
            button.addEventListener("click", function () {{
                if (!window.confirm("Delete this job?")) {{
                    return;
                }}

                const jobId = button.getAttribute("data-job-id");

                fetch(DELETE_URL, {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ job_id: jobId }}),
                }})
                    .then(function (response) {{
                        if (!response.ok) {{
                            throw new Error("Server responded with an error.");
                        }}
                        // server.py has already rebuilt report.html (with
                        // this job gone) by the time it replies, so just
                        // reload the page to show the fresh version -
                        // no manual DOM surgery needed.
                        location.reload();
                    }})
                    .catch(function () {{
                        // Most likely cause: server.py isn't running, so
                        // the fetch couldn't connect at all.
                        alert("Start server.py to enable delete.");
                    }});
            }});
        }});
    </script>
</body>
</html>
"""


# A distinct color per source, purely so the eye can tell boards apart
# at a glance when scanning a long, mixed-source list. Any source not
# listed here (a newly added one that hasn't been given a color yet)
# falls back to SOURCE_BADGE_DEFAULT_COLOR - never a KeyError.
SOURCE_BADGE_COLORS = {
    "remoteok": "#c0392b",
    "remotive": "#2465b0",
    "arbeitnow": "#1b8a3d",
    "himalayas": "#8e44ad",
    "jobicy": "#c77700",
    "themuse": "#117a8b",
    "weworkremotely": "#3d5875",
    "upwork-extension": "#14a800",
}
SOURCE_BADGE_DEFAULT_COLOR = "#555555"


def format_iso_timestamp_date(iso_timestamp):
    """
    Turn an ISO timestamp (with time, e.g. "2026-07-26T14:03:47") into a
    friendly date-only string like "July 26, 2026". Falls back to the
    raw string (or "unknown" if empty) rather than crashing over a
    malformed/missing timestamp.
    """
    if not iso_timestamp:
        return "unknown"
    try:
        return datetime.datetime.fromisoformat(iso_timestamp).strftime("%B %d, %Y")
    except ValueError:
        return iso_timestamp


def build_gap_line_html(entry):
    """
    Build the "Strong match, but missing: ..." line for a single job
    card - see PART 6(a) of the skill-gap-analysis feature. Only shown
    for jobs scoring ABOVE SKILL_GAP_SCORE_THRESHOLD (a genuine
    near-miss is worth analyzing; a low-scoring job's gaps are just
    noise) that actually have at least one gap recorded. Returns "" if
    neither condition holds.
    """
    if entry["score"] <= SKILL_GAP_SCORE_THRESHOLD or not entry.get("gaps"):
        return ""

    quick = [g["text"] for g in entry["gaps"] if g["category"] == "quick"]
    large = [g["text"] for g in entry["gaps"] if g["category"] == "large"]

    parts = []
    if quick:
        parts.append(f'<strong>quick to learn:</strong> {html.escape(", ".join(quick))}')
    if large:
        parts.append(f'<strong>large gap:</strong> {html.escape(", ".join(large))}')
    if not parts:
        return ""

    return f'<p class="gap-line">Strong match, but missing: {" &middot; ".join(parts)}</p>'


def build_daily_job_card_html(job_hash, entry):
    """
    Build the HTML for a single card in the persistent daily_report.html
    list: score, a colored source badge, both date stamps (when the
    posting says it was posted, and when THIS tool first found it),
    title, verdict/flags, a skill-gap line for near-miss jobs (see
    build_gap_line_html()), a prominent "Open job" link, and a "Remove"
    button that permanently deletes this job from JOB_LIST_FILE (see
    server.py's /remove_daily_job). Deliberately no proposal/cover-letter
    section - see run_scoring_and_report()'s draft_proposals docstring
    for why this flow skips that step entirely.
    """
    color = score_to_color(entry["score"])
    badge_color = SOURCE_BADGE_COLORS.get(entry["source"], SOURCE_BADGE_DEFAULT_COLOR)

    posted_display = format_date_heading(entry["date_posted"]) if entry.get("date_posted") else "unknown"
    found_display = format_iso_timestamp_date(entry.get("date_found", ""))

    flags_html = ""
    if entry["flags"]:
        flag_items_html = "".join(f"<li>{html.escape(flag)}</li>" for flag in entry["flags"])
        flags_html = f"""
        <div class="flags-box">
            <div class="flags-label">⚠ Flags</div>
            <ul>{flag_items_html}</ul>
        </div>
        """

    link_html = ""
    if entry["url"]:
        link_html = (
            f'<a class="open-job-button" href="{html.escape(entry["url"])}" '
            f'target="_blank" rel="noopener noreferrer">Open job &rarr;</a>'
        )

    return f"""
    <div class="card" data-job-id="{job_hash}">
        <div class="card-header">
            <span class="score" style="color: {color};">{entry['score']}/100</span>
            <button class="remove-button" data-job-id="{job_hash}" title="Remove from list">&times;</button>
        </div>
        <div class="badges-row">
            <span class="source-badge" style="background-color: {badge_color};">{html.escape(entry['source'])}</span>
        </div>
        <h3 class="job-title">{html.escape(entry['title'])}</h3>
        <p class="date-stamps">Posted: {html.escape(posted_display)} &middot; Found: {html.escape(found_display)}</p>
        <p class="verdict">{html.escape(entry['verdict'])}</p>
        {build_gap_line_html(entry)}
        {flags_html}
        {link_html}
    </div>
    """


def compute_skill_gap_summary(jobs_dict):
    """
    Aggregate skill gaps across every job in `jobs_dict` scoring ABOVE
    SKILL_GAP_SCORE_THRESHOLD - see PART 6(b) of the skill-gap-analysis
    feature: "the market keeps asking for these; focus here first."
    Gaps are grouped by their EXACT text (simple, not fuzzy-matched - two
    postings both saying "Docker" count together, "Docker" and
    "containerization" don't, since we can't reliably tell those are the
    same thing without another LLM call).

    Returns (quick_wins, large_gaps): each a list of (gap_text, count)
    tuples sorted by count descending (most-requested first). quick_wins
    is what to actually act on first per PART 6(b) - a gap that's both
    frequent AND learnable in days/weeks is the highest-value thing to
    pick up next; large_gaps is shown too, for visibility, but isn't
    something a few weekends fixes.
    """
    quick_counts = {}
    large_counts = {}

    for entry in jobs_dict.values():
        if entry["score"] <= SKILL_GAP_SCORE_THRESHOLD:
            continue
        for gap in entry.get("gaps", []):
            counts = quick_counts if gap["category"] == "quick" else large_counts
            key = gap["text"].strip()
            if key:
                counts[key] = counts.get(key, 0) + 1

    quick_wins = sorted(quick_counts.items(), key=lambda kv: -kv[1])
    large_gaps = sorted(large_counts.items(), key=lambda kv: -kv[1])
    return quick_wins, large_gaps


def build_skill_gap_summary_html(jobs_dict):
    """
    Build the "Focus on these skills" section for daily_report.html (see
    compute_skill_gap_summary()) - an aggregate view across every
    near-miss job (score > SKILL_GAP_SCORE_THRESHOLD) in the current
    list, ranked by how often each gap shows up. Returns "" if there are
    no near-miss jobs with recorded gaps at all, so the section simply
    doesn't render rather than showing an empty shell.
    """
    quick_wins, large_gaps = compute_skill_gap_summary(jobs_dict)
    if not quick_wins and not large_gaps:
        return ""

    def build_list_html(items):
        return "".join(
            f"<li>{html.escape(text)} <span class=\"gap-count\">&times;{count}</span></li>"
            for text, count in items
        )

    quick_html = ""
    if quick_wins:
        quick_html = f"""
        <h3>Quick wins (learnable in days/weeks)</h3>
        <ol class="gap-summary-list">{build_list_html(quick_wins)}</ol>
        """

    large_html = ""
    if large_gaps:
        large_html = f"""
        <h3>Bigger investments (years of experience / different domain)</h3>
        <ol class="gap-summary-list">{build_list_html(large_gaps)}</ol>
        """

    return f"""
    <div class="gap-summary-section">
        <h2>🎯 Focus on these skills</h2>
        <p class="gap-summary-intro">Aggregated from every job scoring above
        {SKILL_GAP_SCORE_THRESHOLD} in your list right now - these are the
        gaps that keep costing you a clean match. Quick wins first: a gap
        that's both frequent AND learnable in days/weeks unlocks the most
        near-miss jobs for the least effort.</p>
        {quick_html}
        {large_html}
    </div>
    """


def build_daily_report_html(jobs_dict):
    """
    Build the full HTML page for daily_report.html from the PERSISTENT
    job list (JOB_LIST_FILE's "jobs" dict: hash -> entry - see
    update_persistent_job_list()). Unlike report.html, this list
    ACCUMULATES across every --daily run rather than being regenerated
    from scratch - see the module-level comment on JOB_LIST_FILE.

    Sorted highest-SCORE-first, always - that's the whole point of a fit
    list. Ties are broken by newest-found-first (`date_found` is always a
    precise timestamp, unlike `date_posted`, which some sources don't
    provide), just so equally-scored jobs have a stable, sensible order
    rather than whatever order the dict happened to iterate in.

    Each card has a "Remove" (x) button (server.py's /remove_daily_job)
    that permanently deletes it from JOB_LIST_FILE, and the page has one
    "Clear all" button (server.py's /clear_daily_list) to wipe the whole
    list. No proposal/cover-letter section and no "apply"/"submit"
    button anywhere - see the note on manual applying in the README.
    """
    generated_at_str = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    # Sort by the tie-breaker FIRST, then by score - Python's sort is
    # stable, so ties in score keep the newest-found-first order from
    # this first pass instead of an arbitrary one.
    sorted_items = sorted(jobs_dict.items(), key=lambda item: item[1].get("date_found", ""), reverse=True)
    sorted_items.sort(key=lambda item: item[1]["score"], reverse=True)

    gap_summary_html = build_skill_gap_summary_html(jobs_dict)

    if sorted_items:
        cards_html = "".join(build_daily_job_card_html(job_hash, entry) for job_hash, entry in sorted_items)
    else:
        cards_html = '<p class="empty-state">Your list is empty - check back after the next --fetch/--daily run.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Daily Fresh Job Matches</title>
<style>
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
        background-color: #f4f5f7;
        color: #22262b;
        max-width: 820px;
        margin: 40px auto;
        padding: 0 20px 60px 20px;
        line-height: 1.5;
    }}
    .page-header {{
        margin-bottom: 32px;
    }}
    h1 {{
        font-size: 26px;
        margin: 0 0 6px 0;
    }}
    .generated-at {{
        font-size: 13px;
        color: #777;
        margin: 0 0 4px 0;
    }}
    .manual-note {{
        font-size: 13px;
        color: #8a6215;
        background-color: #fff8e6;
        border-left: 4px solid #d9a441;
        border-radius: 6px;
        padding: 8px 14px;
        margin-top: 14px;
    }}
    .card {{
        background-color: #ffffff;
        border: 1px solid #e2e2e2;
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 15px;
        font-weight: 600;
    }}
    .score {{
        font-size: 21px;
        font-weight: 700;
    }}
    .remove-button {{
        border: 1px solid #e0b4b4;
        border-radius: 6px;
        background-color: #fdf2f2;
        color: #a33;
        font-size: 16px;
        line-height: 1;
        width: 28px;
        height: 28px;
        cursor: pointer;
    }}
    .remove-button:hover {{
        background-color: #fbe4e4;
    }}
    .clear-all-button {{
        margin-top: 12px;
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid #e0b4b4;
        border-radius: 6px;
        background-color: #fdf2f2;
        color: #a33;
        cursor: pointer;
    }}
    .clear-all-button:hover {{
        background-color: #fbe4e4;
    }}
    .badges-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
    }}
    .source-badge {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #ffffff;
        border-radius: 4px;
        padding: 2px 8px;
    }}
    .job-title {{
        font-size: 17px;
        font-weight: 600;
        margin: 10px 0 4px 0;
    }}
    .date-stamps {{
        font-size: 12px;
        color: #888;
        margin: 0 0 10px 0;
    }}
    .verdict {{
        margin: 0 0 10px 0;
        color: #444;
    }}
    .flags-box {{
        background-color: #fff8e6;
        border-left: 4px solid #d9a441;
        border-radius: 6px;
        padding: 10px 16px;
        margin: 10px 0;
        font-size: 13px;
        color: #6b4e14;
    }}
    .flags-label {{
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #8a6215;
        margin-bottom: 4px;
    }}
    .flags-box ul {{
        margin: 0;
        padding-left: 18px;
    }}
    .open-job-button {{
        display: inline-block;
        margin-top: 14px;
        padding: 10px 18px;
        font-size: 14px;
        font-weight: 700;
        color: #ffffff;
        background-color: #1a5fb4;
        border-radius: 6px;
        text-decoration: none;
    }}
    .open-job-button:hover {{
        background-color: #164d92;
    }}
    .empty-state {{
        color: #666;
    }}
    .gap-line {{
        margin: 8px 0 0 0;
        padding: 8px 12px;
        font-size: 13px;
        color: #4a3b6b;
        background-color: #f2eefc;
        border-left: 4px solid #8a63d2;
        border-radius: 6px;
    }}
    .gap-summary-section {{
        background-color: #ffffff;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 0 0 24px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .gap-summary-section h2 {{
        margin: 0 0 8px 0;
    }}
    .gap-summary-section h3 {{
        margin: 16px 0 6px 0;
        font-size: 14px;
        color: #333;
    }}
    .gap-summary-intro {{
        margin: 0;
        font-size: 13px;
        color: #555;
    }}
    .gap-summary-list {{
        margin: 0;
        padding-left: 22px;
        font-size: 13px;
    }}
    .gap-summary-list li {{
        margin: 4px 0;
    }}
    .gap-count {{
        color: #888;
        font-size: 12px;
    }}
</style>
</head>
<body>
    <div class="page-header">
        <h1>ML/AI Job List</h1>
        <p class="generated-at">Last updated {generated_at_str} &middot; {len(sorted_items)} job(s) in your list</p>
        <p class="manual-note">Scores and verdicts only - no cover letters drafted in this
        flow. Open a job's link and apply yourself on the original board;
        nothing here applies automatically. This list is persistent - it
        accumulates across every --fetch/--daily run and only shrinks when
        YOU remove a job below or click "Clear all."</p>
        <button id="clear-all-button" class="clear-all-button">Clear all</button>
    </div>
    {gap_summary_html}
    {cards_html}

    <script>
        // This runs in the browser when daily_report.html is opened.
        // server.py (localhost:8765) must be running for Remove/Clear
        // all to work - see README.md's setup section.
        const REMOVE_URL = "http://localhost:8765/remove_daily_job";
        const CLEAR_URL = "http://localhost:8765/clear_daily_list";

        document.querySelectorAll(".remove-button").forEach(function (button) {{
            button.addEventListener("click", function () {{
                if (!window.confirm("Remove this job from your list?")) {{
                    return;
                }}
                const jobId = button.getAttribute("data-job-id");
                fetch(REMOVE_URL, {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ job_id: jobId }}),
                }})
                    .then(function (response) {{
                        if (!response.ok) {{
                            throw new Error("Server responded with an error.");
                        }}
                        location.reload();
                    }})
                    .catch(function () {{
                        alert("Start src/server.py to enable removing jobs.");
                    }});
            }});
        }});

        document.getElementById("clear-all-button").addEventListener("click", function () {{
            if (!window.confirm("Clear your ENTIRE job list? This cannot be undone.")) {{
                return;
            }}
            fetch(CLEAR_URL, {{ method: "POST" }})
                .then(function (response) {{
                    if (!response.ok) {{
                        throw new Error("Server responded with an error.");
                    }}
                    location.reload();
                }})
                .catch(function () {{
                    alert("Start src/server.py to enable clearing the list.");
                }});
        }});
    </script>
</body>
</html>
"""


def rebuild_daily_report():
    """
    Rebuild daily_report.html directly from the CURRENT JOB_LIST_FILE,
    with NO scoring/fetching side effects - a pure function of whatever
    is already on disk. Used by cli_run_daily() after updating the list,
    and by server.py's /remove_daily_job and /clear_daily_list so the
    page reflects a removal/clear immediately (same pattern as
    build_report() for report.html after a delete).
    """
    job_list = load_job_list()
    daily_report_html = build_daily_report_html(job_list["jobs"])
    with open(DAILY_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(daily_report_html)


def cli_fetch_jobs():
    """
    Handle `python match_llm.py --fetch`: pull jobs from every source in
    sources.SOURCE_FETCHERS (RemoteOK, Remotive, ...), append any new
    ones to jobs.json, and print a per-source summary (found / kept /
    skipped as duplicates). Never touches results.json or report.html -
    scoring only happens in run_scoring_and_report() (see --daily, or a
    plain `python match_llm.py` run afterward).

    Returns the list of newly added job dicts, so cli_run_daily() can
    report on them too without re-fetching.
    """
    jobs = load_jobs(JOBS_JSON_FILE)

    print("Fetching from automatic sources...")
    new_jobs, stats = sources.fetch_all_jobs(jobs)

    if new_jobs:
        jobs.extend(new_jobs)
        save_jobs(jobs)

    print("\n===== Fetch summary =====")
    for source_name, source_stats in stats.items():
        print(
            f"{source_name}: {source_stats['found']} found, "
            f"{source_stats.get('remote_filtered', 0)} dropped (on-site/hybrid), "
            f"{source_stats.get('domain_filtered', 0)} dropped (wrong field/role), "
            f"{source_stats['keyword_matched']} matched keywords, "
            f"{source_stats['active']} still active (not expired), "
            f"{source_stats['duplicates']} already in {JOBS_JSON_FILE}, "
            f"{source_stats['added']} newly added"
        )
    print(f"\nTotal new jobs added: {len(new_jobs)}")

    return new_jobs


def cli_run_daily():
    """
    Handle `python match_llm.py --daily`: the full end-to-end daily
    workflow.
        1. Fetch new jobs from every automatic source (cli_fetch_jobs()).
           These are normal job boards, not Upwork - sources.py keeps
           any job still actively listed (regardless of how long ago it
           was posted) and only drops one a source explicitly marks as
           expired (see sources.posting_status()).
        2. Score any unscored jobs, reusing results.json for everything
           else (run_scoring_and_report(use_cache=True) only ever calls
           Ollama for jobs missing from the cache - once the very first
           backlog is cleared, a normal day is a handful of fresh
           postings across 7 boards, not the whole job history, so this
           finishes quickly). draft_proposals=False here on purpose - no
           cover letters in this flow, see that parameter's docstring on
           run_scoring_and_report().
        3. Add any newly-qualifying jobs to the PERSISTENT job list (see
           update_persistent_job_list() - score above FIT_SCORE_THRESHOLD,
           not a role-type mismatch, not previously removed by you).
        4. Rebuild daily_report.html from the full persistent list
           (rebuild_daily_report()) - this is NOT "just today's jobs", it
           ACCUMULATES every qualifying job ever found, across every run,
           until you remove it yourself.
        5. Print a short terminal summary.

    This NEVER submits anything anywhere - see the README's note on why
    applying stays a manual, human step.
    """
    print("=== Step 1/4: fetching new, fresh jobs ===")
    cli_fetch_jobs()

    print("\n=== Step 2/4: scoring today's new jobs (no proposal drafting - scores only) ===")
    run_scoring_and_report(use_cache=True, draft_proposals=False)

    print("\n=== Step 3/4: updating your persistent job list ===")
    job_list, added, deduped = update_persistent_job_list()
    print(f"Added {added} newly-qualifying job(s) (score > {FIT_SCORE_THRESHOLD}, real ML/AI engineering fit).")
    if deduped:
        print(f"Removed {deduped} duplicate(s) that shared a URL with another entry (kept the higher-scoring one).")

    print("\n=== Step 4/4: writing daily_report.html ===")
    rebuild_daily_report()

    total = len(job_list["jobs"])
    print(f"\n===== Your job list: {total} job(s), highest score first (see {DAILY_REPORT_FILE}) =====")
    by_score = sorted(job_list["jobs"].values(), key=lambda entry: entry["score"], reverse=True)
    for entry in by_score:
        print(f"[{entry['score']}/100] {entry['title']}  ({entry['source']})")
        if entry["url"]:
            print(f"   {entry['url']}")

    print(f"\nHTML digest saved to {DAILY_REPORT_FILE}")
    print("Open each link and apply yourself - nothing is auto-applied and no proposals were drafted in this flow.")


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match_llm.py`), not
# when it's imported by another file."
def cli_list_jobs():
    """
    Handle `python match_llm.py --list`: print every job in jobs.json
    with a 1-based index number and its title, so the user knows which
    number to pass to --delete.
    """
    jobs = load_jobs(JOBS_JSON_FILE)

    if not jobs:
        print(f"No jobs found in {JOBS_JSON_FILE}.")
        return

    print(f"{len(jobs)} job(s) in {JOBS_JSON_FILE}:\n")
    for index, job in enumerate(jobs, start=1):
        title, _ = extract_title_and_metadata(job.get("text", ""))
        date_added = job.get("date_added", "?")
        print(f"{index}. [{date_added}] {title}")


def cli_delete_job(job_number):
    """
    Handle `python match_llm.py --delete N`: remove job number N (as
    shown by --list) from jobs.json, and also remove its cached result
    from results.json if one exists, so a deleted job doesn't linger in
    either file.
    """
    jobs = load_jobs(JOBS_JSON_FILE)

    if job_number < 1 or job_number > len(jobs):
        print(f"No job #{job_number} - there are only {len(jobs)} job(s). Use --list to see them.")
        return

    # Lists are 0-indexed but we show/accept 1-based numbers to the user.
    removed_job = jobs.pop(job_number - 1)
    save_jobs(jobs)

    title, _ = extract_title_and_metadata(removed_job.get("text", ""))

    results = load_results()
    removed_hash = hash_job_text(removed_job.get("text", ""))
    had_cached_result = removed_hash in results
    if had_cached_result:
        del results[removed_hash]
        save_results(results)

    print(f"Deleted job #{job_number}: {title}")
    if had_cached_result:
        print("Also removed its cached result from results.json.")


def build_report():
    """
    Rebuild report.html directly from the CURRENT jobs.json and cached
    results.json, WITHOUT calling Ollama. Jobs that don't have a cached
    result yet are shown as "Not scored yet" cards instead of being
    scored on the spot.

    This is the single source of truth for turning jobs.json + the
    results cache into a grouped report, used both by the normal scoring
    run below (after it finishes scoring/caching) and by server.py right
    after a job is deleted - so report.html reflects jobs.json
    immediately, without needing a full rescoring run.

    Returns (results_by_date, sorted_dates) so callers that also want to
    print a terminal summary (like run_scoring_and_report()) don't have
    to rebuild the same grouping a second time.
    """
    jobs = load_jobs(JOBS_JSON_FILE)
    results_cache = load_results()

    # Group jobs by date_added, so today's jobs stay visually separate
    # from older ones in the HTML report.
    results_by_date = {}

    for job in jobs:
        job_text = job.get("text", "")
        date_added = job.get("date_added") or "Unknown date"
        title, metadata = extract_title_and_metadata(job_text)
        job_hash = hash_job_text(job_text)

        cached = results_cache.get(job_hash)
        if cached:
            score = cached["score"]
            verdict = cached["verdict"]
            proposal = cached.get("proposal")
            # .get(..., []) so results cached before FLAGS existed don't
            # crash the report - they just show no flags.
            flags = cached.get("flags", [])
        else:
            # No cached result for this job - show it as unscored rather
            # than calling Ollama here (that only happens in
            # run_scoring_and_report(), never as a side effect of
            # rebuilding the report).
            score = None
            verdict = 'Not scored yet - run "python match_llm.py" to score this job.'
            proposal = None
            flags = []

        entry = {
            "score": score,
            "verdict": verdict,
            "title": title,
            "metadata": metadata,
            "proposal": proposal,
            "flags": flags,
            "job_id": job_hash,  # lets the HTML report's delete button identify this job
        }
        results_by_date.setdefault(date_added, []).append(entry)

    # Within each date, rank scored jobs highest-first; unscored jobs
    # (score is None) sort to the bottom of their date section rather
    # than crashing a plain numeric sort.
    for date_added, entries in results_by_date.items():
        entries.sort(key=lambda entry: (entry["score"] is None, -(entry["score"] or 0)))
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

    # Newest date first. "YYYY-MM-DD" strings sort correctly as plain
    # text, so a simple reverse-sorted() is enough here.
    sorted_dates = sorted(results_by_date.keys(), reverse=True)

    report_html = build_html_report(results_by_date, sorted_dates)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_html)

    return results_by_date, sorted_dates


def run_scoring_and_report(use_cache, draft_proposals=True):
    """
    The main pipeline: score every job in jobs.json (reusing cached
    results from results.json unless `use_cache` is False), then hand
    off to build_report() to group everything by date, write
    report.html, and print a terminal summary.

    `draft_proposals` controls whether high-scoring jobs also get a
    drafted cover letter (an extra, slower Ollama call per job). The
    manual Upwork extension flow (a plain `python match_llm.py` run)
    wants that - see report.html's proposal boxes. --daily's automatic
    job-board flow explicitly does NOT: with potentially dozens of fresh
    postings across 7 sources every run, drafting a proposal for every
    one of them would turn a "few seconds" daily check back into a
    multi-minute one, and the user reviews/writes their own note before
    applying anyway - see cli_run_daily().
    """
    jobs = load_jobs(JOBS_JSON_FILE)
    results_cache = load_results()

    print(
        f"Loaded {len(jobs)} jobs from {JOBS_JSON_FILE}. Scoring each with "
        f"{LLM_PROVIDER}:{MODEL_NAME} (fallback: ollama:{FALLBACK_OLLAMA_MODEL})...\n"
    )

    for job in jobs:
        job_text = job.get("text", "")
        title, _ = extract_title_and_metadata(job_text)
        job_hash = hash_job_text(job_text)

        cached_entry = results_cache.get(job_hash)
        # Reuse a cached result only if it's present, the caller didn't
        # ask for a fresh re-score (--fresh), AND it has a "model" field.
        # An entry missing "model" is a LEGACY entry from before this
        # field existed (or from a mixed-model cache before a provider
        # switch) - we can't trust which backend actually produced it,
        # so it's treated the same as "not scored yet" and gets a fresh
        # score here (which will then carry a proper "model" label).
        if use_cache and cached_entry is not None and "model" in cached_entry:
            print(f"Reused cached: {title}")
            continue

        if cached_entry is not None:
            print(f"Scoring new (legacy cache entry, no model recorded): {title}")
        else:
            print(f"Scoring new: {title}")

        raw_reply, model_label = score_job(job_text)
        computed_score = compute_score_from_components(raw_reply)

        # The model occasionally skips the (yes)/(partial)/(no) labels
        # entirely, which leaves us nothing reliable to compute a score
        # from. At temperature=0 a retry with the SAME prompt would just
        # reproduce the same malformed reply, so give it one retry with
        # a little randomness instead, purely to get a differently-shaped
        # (hopefully compliant) answer.
        if computed_score is None:
            print(f"  (retrying - missing component labels: {title})")
            raw_reply, model_label = score_job(job_text, temperature=0.2)
            computed_score = compute_score_from_components(raw_reply)

        score, verdict = parse_score_and_verdict(raw_reply)
        flags = parse_flags(raw_reply)
        gaps = parse_gaps(raw_reply)

        # Prefer the score computed deterministically from the
        # COMPONENTS classification over the model's own (unreliable)
        # arithmetic - see compute_score_from_components() for why.
        # Falls back to the model's SCORE line if COMPONENTS still
        # couldn't be parsed after the retry.
        if computed_score is not None:
            score = computed_score

        # Code-level gates - role type (hard cap) and seniority (soft
        # penalty) - see apply_score_adjustments() above for why these
        # are enforced in code rather than trusted to the model's own
        # judgment. Location is a flag only, added to `flags` below and
        # never affecting the score. The full-time-employee gate is NOT
        # applied here at all any more - it's a hard EXCLUSION from the
        # persistent list (see update_persistent_job_list()), not a
        # score penalty or a flag, so an ineligible job's raw score stays
        # untouched in the cache (useful if eligibility ever changes).
        domain_fit = parse_domain_fit(raw_reply)
        score = apply_score_adjustments(score, domain_fit, title, job_text)

        location_warning = location_flag(job_text)
        if location_warning:
            flags = flags + [location_warning]

        # Only draft a proposal for high-scoring jobs - it's not worth
        # spending the model's (or the client's) time on a poor fit. And
        # only if the caller actually wants proposals at all (--daily
        # doesn't - see the draft_proposals docstring above).
        proposal = None
        if draft_proposals and score >= PROPOSAL_SCORE_THRESHOLD:
            proposal = draft_proposal(job_text)

        results_cache[job_hash] = {
            "title": title,
            "score": score,
            "verdict": verdict,
            "proposal": proposal,
            "flags": flags,
            "gaps": gaps,
            "model": model_label,
        }

        # Save after EVERY job, not just once at the end - a full run can
        # take a long time, and without this, an interrupted run (crash,
        # Ctrl+C, killed process) would lose every score computed so
        # far, forcing a full re-score from scratch on the next attempt.
        save_results(results_cache)

    # Everything is already saved incrementally above; this final call is
    # a harmless no-op when the loop ran to completion (results_cache is
    # already on disk), and a no-op when it was empty (nothing to save).
    save_results(results_cache)

    results_by_date, sorted_dates = build_report()

    print("\n===== Ranked Job Matches (grouped by date) =====\n")
    for date_added in sorted_dates:
        print(f"--- {format_date_heading(date_added)} ---")
        for entry in results_by_date[date_added]:
            score_display = "Not scored yet" if entry["score"] is None else entry["score"]
            print(f"#{entry['rank']} - SCORE: {score_display} - {entry['title']}")
            if entry["metadata"]:
                print(f"   {entry['metadata']}")
            print(f"   Verdict: {entry['verdict']}")
            if entry["flags"]:
                print(f"   Flags: {'; '.join(entry['flags'])}")
            if entry["proposal"]:
                print("\n   Drafted proposal:")
                print(f"   {entry['proposal']}")
            print()
        print()

    print(f"HTML report saved to {REPORT_FILE}")


if __name__ == "__main__":
    # Job titles/descriptions can contain arbitrary Unicode (accented
    # names, em dashes, etc.) - especially now that sources.py pulls jobs
    # in automatically instead of everything being hand-typed/pasted.
    # Some terminals (observed here: a Windows console defaulting to the
    # cp1256 codepage) can't encode certain of those characters, which
    # crashes a plain print() mid-run. Reconfiguring stdout/stderr to
    # UTF-8 with errors="replace" makes every print() in this file safe
    # regardless of the terminal's own codepage, without changing what
    # gets printed on a terminal that already handles UTF-8 fine.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # very old Python without TextIOWrapper.reconfigure() - just skip

    cli_args = sys.argv[1:]

    if "--fetch" in cli_args:
        cli_fetch_jobs()
    elif "--daily" in cli_args:
        cli_run_daily()
    elif "--list" in cli_args:
        cli_list_jobs()
    elif "--delete" in cli_args:
        delete_flag_index = cli_args.index("--delete")
        try:
            job_number_to_delete = int(cli_args[delete_flag_index + 1])
        except (IndexError, ValueError):
            print("Usage: python match_llm.py --delete <job number> (see --list for numbers)")
        else:
            cli_delete_job(job_number_to_delete)
    else:
        # --fresh ignores the results.json cache and re-scores every
        # job with Ollama, regardless of whether it's been scored before.
        use_cache = "--fresh" not in cli_args
        run_scoring_and_report(use_cache)
