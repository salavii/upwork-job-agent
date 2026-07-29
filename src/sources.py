"""
sources.py

Automatic job discovery from public, official job-board APIs/feeds - a
SECOND, automatic source of jobs alongside the existing Chrome-extension +
server.py flow (which this file does not touch or depend on in any way).

Only documented, public, official endpoints are used here - no scraping,
no headless browsers, no login, no bypassing rate limits:
- RemoteOK        - https://remoteok.com/api                       (public JSON API)
- Remotive        - https://remotive.com/api/remote-jobs           (public JSON API)
- Arbeitnow       - https://www.arbeitnow.com/api/job-board-api    (public JSON API)
- Himalayas       - https://himalayas.app/jobs/api                 (public JSON API)
- Jobicy          - https://jobicy.com/api/v2/remote-jobs          (public JSON API)
- The Muse        - https://www.themuse.com/api/public/jobs        (public JSON API)
- WeWorkRemotely  - https://weworkremotely.com/categories/*.rss    (public RSS feed)

Sources considered and skipped for NOT having a genuinely public/free API:
- Adzuna requires a registered API key (not a truly public/anonymous
  endpoint) - skipped.
- GitHub Jobs was discontinued by GitHub.
- LinkedIn/Indeed have no public job-search API for third parties.

Adding another source later is just: write one `fetch_<name>()` function
following the same (jobs, funnel_stats) contract as the ones below, then
add it to SOURCE_FETCHERS at the bottom of this file. Nothing else in
match_llm.py needs to change.

Every normalized job is shaped EXACTLY like the entries server.py already
writes to jobs.json, plus three extra fields:
    {
        "text": "title — company\\n\\nlocation/type\\n\\ndescription",
        "date_added": "YYYY-MM-DD",
        "saved_at": "<ISO timestamp>",
        "source": "remoteok" | "remotive" | "arbeitnow" | "himalayas" | "jobicy" | "themuse" | "weworkremotely",
        "url": "https://... (link to the original posting)",
        "freshness": "active" | "expired",
    }
The extra keys are additive - match_llm.py and server.py's existing code
only ever reads "text"/"date_added" from a job dict, so older jobs (and
jobs saved by the extension, which never set these keys) keep working
unchanged.
"""

import datetime
import email.utils
import hashlib
import html
import re
import time

import requests

# A real, identifying User-Agent and a bounded timeout, so a slow or
# unreachable API can't hang --fetch/--daily forever, and so we're not
# just another anonymous client hammering someone's public API.
REQUEST_HEADERS = {
    "User-Agent": "ali-job-matcher-personal-script/1.0 (personal use, low volume)"
}
REQUEST_TIMEOUT_SECONDS = 15

# One retry on a transient network failure (timeout, connection reset,
# 5xx) before giving up on a source for this run - see _get() below.
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2

# These are normal job boards, not Upwork - what matters is whether a
# posting is STILL OPEN, not how recently it was posted (no "first to
# apply wins" dynamic here). There is deliberately NO age-based cutoff
# any more - a good job posted a week ago that's still listed is exactly
# as valid as one posted today. Every source's API only returns
# currently-listed postings in the first place, so a job is kept unless
# the source gives concrete evidence it has expired - see
# posting_status() below (currently only Himalayas exposes an
# "expiryDate"; every other source has no such signal, so nothing there
# is ever dropped for "staleness").

# Only jobs whose title+description mention at least one of these are
# kept - broad job-board feeds are mostly irrelevant noise (sales,
# WordPress, customer support, ...) for an ML-focused profile. This is a
# config constant on purpose, so it's easy to tune without touching the
# fetch logic. Terms short enough to appear inside unrelated words ("AI",
# "ML", "RAG", "NLP", "LLM") are wrapped in \b word boundaries so "email"
# or "storage" can't match.
KEYWORD_PATTERNS = [
    r"machine learning",
    r"\bML\b",
    r"\bML engineer\b",
    r"\bML developer\b",
    r"deep learning",
    r"\bAI\b",
    r"\bLLM\b",
    r"large language model",
    r"\bNLP\b",
    r"natural language processing",
    r"\btransformers?\b",
    r"\bRAG\b",
    r"retrieval[- ]augmented generation",
    r"computer vision",
    r"\bCNN\b",
    r"image classification",
    r"PyTorch",
    r"TensorFlow",
    r"\bKeras\b",
    r"fine[- ]?tuning",
    r"\bLoRA\b",
    r"\bPEFT\b",
    r"prompt engineering",
    r"hugging face",
    r"scikit-learn",
    r"\bOpenCV\b",
    r"\bPython\b",
    r"\bFastAPI\b",
    r"\bChromaDB\b",
    r"data scientist",
    r"model training",
    r"embeddings?",
    r"vector database",
    r"\bAI trainer\b",
    r"\bLLM evaluator\b",
    r"model evaluation",
    r"\bred[\s-]?team(ing)?\b",
    r"\bRLHF\b",
    r"reinforcement learning from human feedback",
    r"\bannotation\b",
    r"\bdata annotator\b",
]
RELEVANT_KEYWORDS_PATTERN = re.compile("|".join(KEYWORD_PATTERNS), re.IGNORECASE)

# STEP 4 (negative-keyword gate) - a job whose TITLE contains one of these
# is almost certainly the wrong field entirely (surveying/geospatial work
# that happens to use "AI" as a buzzword) or a non-technical business
# role (recruiting/sales/accounting/support), even at an otherwise
# ML/AI-focused company. Checked title-only, same reasoning as
# is_relevant()'s title check and match_llm.py's ENGINEERING_TITLE_PATTERN
# backstop: a title clearly signaling a real engineering/research role
# (see CLEAR_ENGINEERING_TITLE_PATTERN below) always wins over an
# incidental domain word - e.g. "Founding AI Engineer, Tax Startup" stays
# in (it's a genuine ML engineering role that happens to work in the tax
# domain), but "Tax Accountant" or "Payroll Specialist" does not.
NEGATIVE_DOMAIN_PATTERNS = [
    r"\bgeodesy\b",
    r"\bGIS\b",
    r"\bphotogrammetry\b",
    r"remote sensing",
    r"\bsurveying\b",
    r"\blidar\b",
    r"drone pilot(ing)?",
    r"\bpayroll\b",
    r"\bHR\b",
    r"human resources",
    r"\brecruiting\b",
    r"\brecruiter\b",
    r"\bsales\b",
    r"\bmarketing\b",
    r"\baccounting\b",
    r"\baccountant\b",
    r"\btax\b",
    r"customer support",
    r"customer service",
]
NEGATIVE_DOMAIN_PATTERN = re.compile("|".join(NEGATIVE_DOMAIN_PATTERNS), re.IGNORECASE)

CLEAR_ENGINEERING_TITLE_PATTERN = re.compile(
    r"\b(engineer(ing)?|scientist|developer|research(er)?)\b", re.IGNORECASE
)


def is_wrong_domain(text):
    """
    STEP 4 - True if `text`'s title reads as the wrong field (GIS/
    geospatial/surveying work) or a non-technical business role
    (recruiting/sales/marketing/accounting/support), even though it may
    mention "AI" elsewhere. Title-only, with a clear-engineering-title
    override (see NEGATIVE_DOMAIN_PATTERN's comment above) so a genuine
    ML/AI engineering role isn't dropped just because its product domain
    happens to touch one of these words (e.g. an AI tax-automation
    startup's "Founding AI Engineer" posting).
    """
    title = text.split("\n", 1)[0]
    if CLEAR_ENGINEERING_TITLE_PATTERN.search(title):
        return False
    return bool(NEGATIVE_DOMAIN_PATTERN.search(title))


# STEP 2 (remote-only gate) - keep a job only if it's genuinely remote;
# freelance/contract work counts too, since that's remote-by-nature work
# for a client anywhere. See classify_remote_status() below for how these
# combine with each source's own remote signal (or lack of one).
REMOTE_POSITIVE_PATTERN = re.compile(
    r"\bfully remote\b|\bremote[\s-]?first\b|\b100%\s*remote\b"
    r"|\bwork from anywhere\b|\bremote\s*[-–]?\s*(eu|europe|usa|us|uk|global|worldwide)\b"
    r"|\bworldwide\b|\banywhere\b|\bfreelance\b|\bcontract\b|\bremote\b",
    re.IGNORECASE,
)
REMOTE_NEGATIVE_PATTERN = re.compile(
    r"\bhybrid\b|\bon[\s-]?site\b|\bonsite\b|\bin[\s-]?office\b"
    r"|\b\d+\s*days?\s*(a|per)\s*week\s*(in|at)\s*(the\s*)?office\b",
    re.IGNORECASE,
)

# Terms in a location field that mean "no specific place, no signal
# either way" - not enough alone to call something remote (that's what
# REMOTE_POSITIVE_PATTERN is for) but not a real place name either.
_GENERIC_LOCATION_TERMS = {"", "remote", "worldwide", "anywhere", "global", "n/a", "unknown"}

# RemoteOK, Remotive, Himalayas, Jobicy, and WeWorkRemotely are all
# remote-jobs-ONLY boards by definition - every listing on them is remote
# work, even though their location/region fields describe geographic
# ELIGIBILITY ("USA Only", "Worldwide") rather than remote-vs-onsite
# status. Arbeitnow and The Muse are general job boards that list both
# remote and on-site/hybrid roles side by side, so they need real
# filtering - see classify_remote_status()'s callers below.
REMOTE_FIRST_SOURCES = {"remoteok", "remotive", "himalayas", "jobicy", "weworkremotely"}


def classify_remote_status(location_or_type, description, remote_hint=None, assume_remote_board=False):
    """
    STEP 2 - classify a posting as "remote", "excluded" (on-site/hybrid),
    or "unconfirmed" (no reliable signal either way - kept, but tagged
    for a manual look, per the user's explicit instruction not to drop
    genuinely ambiguous postings).

    `remote_hint` is a per-JOB, source-provided boolean when the raw API
    exposes one (currently only Arbeitnow's "remote" field), and
    `assume_remote_board` is a per-SOURCE flag for boards that are
    remote-only by definition (see REMOTE_FIRST_SOURCES). Either one is
    an AUTHORITATIVE "yes, this is remote" signal - trusted without
    needing the word "remote" to literally appear anywhere. Only the
    LOCATION field itself (not the full description) can override that
    trust, since a long description commonly contains generic company-
    benefits boilerplate ("our hybrid work culture...") that has nothing
    to do with whether THIS specific listing is remote - checking the
    whole description against an authoritatively-remote listing produced
    false exclusions of postings explicitly titled e.g. "(Remote - U.S.)"
    whose body happened to mention "hybrid" once, in a benefits
    paragraph.

    Falls back to text matching for everything else (The Muse, and
    Arbeitnow when its "remote" field is missing/None): a positive phrase
    ("fully remote", "remote EU", "freelance", "contract", ...) keeps it;
    a negative phrase ("hybrid", "on-site", "3 days a week in the
    office") anywhere in the posting excludes it; a specific place name
    with neither reads as an on-site posting and is excluded; a blank/
    generic location with neither is genuinely unknown and is kept as
    "unconfirmed".
    """
    if remote_hint is False:
        return "excluded"

    if remote_hint is True or assume_remote_board:
        if REMOTE_NEGATIVE_PATTERN.search(location_or_type):
            return "excluded"
        return "remote"

    combined = f"{location_or_type}\n{description}"

    if REMOTE_NEGATIVE_PATTERN.search(combined):
        return "excluded"
    if REMOTE_POSITIVE_PATTERN.search(combined):
        return "remote"

    if location_or_type.strip().lower() in _GENERIC_LOCATION_TERMS:
        return "unconfirmed"
    return "excluded"  # a specific place name, no remote mention


def _apply_remote_gate(location_or_type, description, remote_hint=None, assume_remote_board=False):
    """
    Shared by every fetch_*() below: runs classify_remote_status() and
    returns (keep, location_or_type) - `location_or_type` comes back
    annotated with "[remote-unconfirmed]" when the status is ambiguous,
    so the tag rides along in the job's saved "text" (the simplest way to
    surface it without this file needing to touch match_llm.py's report
    rendering at all - see the module docstring on additive fields).
    `keep` is False only for "excluded" - both "remote" and "unconfirmed"
    are kept, per the STEP 2 instruction to never drop a genuinely
    ambiguous posting.
    """
    remote_status = classify_remote_status(
        location_or_type, description, remote_hint=remote_hint, assume_remote_board=assume_remote_board
    )
    if remote_status == "excluded":
        return False, location_or_type
    if remote_status == "unconfirmed":
        location_or_type = f"{location_or_type} [remote-unconfirmed]".strip()
    return True, location_or_type

_TAG_PATTERN = re.compile(r"<[^>]+>")


def is_relevant(text):
    """
    True if `text` looks like a genuine ML/AI-related role, not just a
    posting that name-drops "AI" once in passing (a buzzword like "AI
    fluency", or a company name like "Vet Vision AI"). A single keyword
    hit anywhere in a long job description turned out to let through a
    lot of unrelated roles (accounting, customer success, sales) that
    happened to mention "AI" exactly once - so this requires EITHER:
      - a keyword match in the TITLE (the text's first line) - a title
        like "Machine Learning Engineer" or "Senior AI/ML Engineer" is a
        strong, direct signal on its own, OR
      - at least two DISTINCT keywords matched anywhere in the text - one
        passing mention isn't enough, but a posting that uses several
        different ML/AI terms almost certainly really is one.
    """
    title = text.split("\n", 1)[0]
    if RELEVANT_KEYWORDS_PATTERN.search(title):
        return True

    distinct_matches = {match.group(0).lower() for match in RELEVANT_KEYWORDS_PATTERN.finditer(text)}
    return len(distinct_matches) >= 2


# A posting that says there's no open role right now isn't a job at
# all (e.g. a stale "careers" page snapshot some boards pick up) - drop
# it at fetch time so it never reaches jobs.json. match_llm.py has its
# own copy of this same pattern (DEAD_POSTING_PATTERN) for catching
# anything that slipped in before this filter existed - keep both in
# sync if this ever changes.
DEAD_POSTING_PATTERN = re.compile(
    r"don'?t currently have any open (roles?|positions?)"
    r"|no (current|open) (openings|positions|roles)"
    r"|not currently hiring",
    re.IGNORECASE,
)


def is_dead_posting(text):
    """True if `text` reads like "we have no open roles" rather than an actual job."""
    return bool(DEAD_POSTING_PATTERN.search(text))


def strip_html(raw_html):
    """
    Turn an HTML job description (every source here returns HTML, not
    plain text) into plain text: drop tags, unescape entities like
    "&amp;", and collapse the extra whitespace tags leave behind.
    """
    text = _TAG_PATTERN.sub(" ", raw_html or "")
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _get(url, params=None):
    """
    GET `url` with our standard headers/timeout, retrying once on a
    transient network failure (timeout, connection error, 5xx) before
    giving up. Raises requests.RequestException on final failure - every
    fetch_*() below catches that in its own try/except, so one source's
    outage can never crash the whole --fetch/--daily run.
    """
    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(
                url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_error


def _parse_iso_datetime(date_str):
    """
    Parse an ISO-ish date string into a timezone-aware datetime, or None
    if it's missing/unparseable. Every fetch_*() below normalizes ITS
    source's native date format (Unix epoch, RFC 2822, ...) into an ISO
    string BEFORE calling build_normalized_job(), so this is the only
    date-parsing path the rest of the module needs to worry about.
    """
    if not date_str:
        return None
    try:
        cleaned = date_str.strip().replace("Z", "+00:00")
        parsed = datetime.datetime.fromisoformat(cleaned)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def _epoch_to_iso(epoch_value):
    """
    Convert a Unix timestamp (int/float/numeric string, in SECONDS) into
    an ISO 8601 string - Arbeitnow's "created_at" and Himalayas'
    "pubDate" are both raw epoch seconds, not ISO strings.
    """
    if epoch_value is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(float(epoch_value), tz=datetime.timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _rfc2822_to_iso(date_str):
    """
    Convert an RFC 2822 date string (e.g. "Tue, 30 Jun 2026 20:34:13
    +0000", WeWorkRemotely's RSS <pubDate> format) into an ISO 8601
    string.
    """
    if not date_str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.isoformat()


def parse_date_posted(date_str):
    """
    Turn an ISO-ish date string into "YYYY-MM-DD", or None if it's
    missing/unparseable - callers fall back to today's date in that case
    rather than crashing over one malformed timestamp.
    """
    parsed = _parse_iso_datetime(date_str)
    return parsed.strftime("%Y-%m-%d") if parsed else None


def posting_status(expiry_timestamp=None):
    """
    Classify whether a posting should be considered currently active:
      - "expired": the source gave a concrete expiry timestamp and it's
        already in the past - drop it, it's not a real open job anymore.
      - "active":  everything else - no expiry info was given (the vast
        majority of postings; these APIs only return currently-listed
        jobs to begin with), or an expiry timestamp that's still in the
        future.

    `expiry_timestamp` is a raw Unix timestamp (seconds) if the source
    provides one (currently only Himalayas' "expiryDate" - see
    fetch_himalayas()), or None otherwise.
    """
    if expiry_timestamp:
        try:
            expires_at = datetime.datetime.fromtimestamp(float(expiry_timestamp), tz=datetime.timezone.utc)
        except (TypeError, ValueError, OSError):
            return "active"
        if expires_at < datetime.datetime.now(datetime.timezone.utc):
            return "expired"

    return "active"


def hash_job_text(job_text):
    """
    Identical formula to match_llm.py's hash_job_text() (and server.py's
    job_text_hash()): strip, encode UTF-8, SHA-256, hex digest. This MUST
    stay in lockstep with those so dedup against jobs.json/results.json
    actually works - duplicated here on purpose (same pattern server.py
    already uses) instead of importing match_llm, which would create a
    circular import (match_llm imports this module for --fetch/--daily).
    """
    return hashlib.sha256(job_text.strip().encode("utf-8")).hexdigest()


_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_company_title_key(job_text):
    """
    Build a normalized "title::company" dedup key from a job's "text"
    field (see build_normalized_job()'s "title — company" first line).
    Used ALONGSIDE the exact-hash dedup (hash_job_text()) to catch
    near-duplicate re-fetches of the SAME real posting: a board can
    re-serve a job with a slightly different description snapshot
    between fetches (whitespace, minor formatting), which produces a
    different exact hash for what is obviously the same posting - this
    key catches that by ignoring everything except the title and company,
    lowercased and stripped of punctuation/whitespace.

    Returns None if the text doesn't have the "title — company" format
    at all (e.g. a manually-saved Upwork job, which never uses this
    separator) - those are only deduped by exact hash, same as before.
    """
    first_line = job_text.strip().splitlines()[0] if job_text.strip() else ""
    if " — " not in first_line:
        return None
    title, _, company = first_line.partition(" — ")
    normalize = lambda s: _NON_ALNUM_PATTERN.sub("", s.lower())
    return f"{normalize(title)}::{normalize(company)}"


def build_normalized_job(
    *, title, company, location_or_type, description, url, source, date_posted=None, expiry_timestamp=None
):
    """
    Build one job dict in the exact shape jobs.json expects (see the
    module docstring above). "text" is built as
    title + company + location/type + full description, matching what
    match_llm.py's extract_title_and_metadata() expects (title = first
    line). `date_posted` must already be an ISO-ish string (or None) -
    each fetch_*() converts its source's native date format before
    calling this. `expiry_timestamp` is a raw Unix timestamp (seconds)
    if the source provides one - see posting_status().
    """
    title = (title or "Untitled").strip()
    company = (company or "Unknown company").strip()
    location_or_type = (location_or_type or "").strip()
    description = (description or "").strip()

    text = f"{title} — {company}\n\n{location_or_type}\n\n{description}".strip()

    date_added = parse_date_posted(date_posted) or datetime.datetime.now().strftime("%Y-%m-%d")

    return {
        "text": text,
        "date_added": date_added,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "url": url or "",
        "freshness": posting_status(expiry_timestamp),
    }


def _empty_funnel():
    return {"found": 0, "keyword_matched": 0, "remote_filtered": 0, "domain_filtered": 0}


def fetch_remoteok():
    """
    Fetch current listings from RemoteOK's public JSON API
    (https://remoteok.com/api - no API key required).

    The API returns a JSON array whose FIRST element is a legal/notice
    blob (no "position"/"company" keys), not a real job - skipped below.

    Returns (jobs, funnel_stats): `jobs` is the list of normalized jobs
    that passed the keyword-relevance filter (RemoteOK exposes no expiry
    info, so nothing is ever dropped here for being "stale" - see the
    module-level note on posting_status()). `funnel_stats` is
    {"found": N, "keyword_matched": M} - N is how many real listings the
    API returned in total, M is how many matched a keyword - used for the
    --fetch summary. Returns ([], funnel) with zeroed counts if the
    request fails for any reason - one source failing must never crash
    the run.
    """
    try:
        response = _get("https://remoteok.com/api")
        raw_jobs = response.json()
    except (requests.RequestException, ValueError) as error:
        print(f"[remoteok] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = 0
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for raw_job in raw_jobs:
        if "position" not in raw_job or "company" not in raw_job:
            continue  # the legal-notice entry, or a malformed row

        found += 1

        location = raw_job.get("location") or "Remote"
        job_type = ", ".join(raw_job.get("tags", []) or [])
        location_or_type = f"{location}" + (f" | {job_type}" if job_type else "")
        description = strip_html(raw_job.get("description", ""))

        # RemoteOK is a remote-jobs-only board - see REMOTE_FIRST_SOURCES.
        remote_ok, location_or_type = _apply_remote_gate(
            location_or_type, description, assume_remote_board=True
        )
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=raw_job.get("position", ""),
            company=raw_job.get("company", ""),
            location_or_type=location_or_type,
            description=description,
            url=raw_job.get("url") or raw_job.get("apply_url") or "",
            source="remoteok",
            date_posted=raw_job.get("date"),
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


def fetch_remotive():
    """
    Fetch current listings from Remotive's public JSON API
    (https://remotive.com/api/remote-jobs - no API key required).

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    try:
        response = _get("https://remotive.com/api/remote-jobs")
        raw_jobs = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as error:
        print(f"[remotive] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = len(raw_jobs)
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for raw_job in raw_jobs:
        location_or_type = f"{raw_job.get('candidate_required_location', 'Remote')} | {raw_job.get('job_type', '')}".strip(
            " |"
        )
        description = strip_html(raw_job.get("description", ""))

        # Remotive is a remote-jobs-only board - see REMOTE_FIRST_SOURCES.
        remote_ok, location_or_type = _apply_remote_gate(
            location_or_type, description, assume_remote_board=True
        )
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("company_name", ""),
            location_or_type=location_or_type,
            description=description,
            url=raw_job.get("url", ""),
            source="remotive",
            date_posted=raw_job.get("publication_date"),
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


def fetch_arbeitnow():
    """
    Fetch current listings from Arbeitnow's public job board API
    (https://www.arbeitnow.com/api/job-board-api - no API key required,
    page 1 = most recently created according to the API's own docs).

    Note: "created_at" is a Unix timestamp in SECONDS, not an ISO
    string - converted via _epoch_to_iso() before freshness/date
    handling.

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    try:
        response = _get("https://www.arbeitnow.com/api/job-board-api")
        raw_jobs = response.json().get("data", [])
    except (requests.RequestException, ValueError) as error:
        print(f"[arbeitnow] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = len(raw_jobs)
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for raw_job in raw_jobs:
        remote_flag = raw_job.get("remote")
        location = raw_job.get("location") or ""
        # Arbeitnow can set BOTH "location" (a real city, e.g. "Berlin")
        # AND "remote": true at the same time (a remote role anchored to
        # a city/country for tax/timezone reasons) - append rather than
        # overwrite, so the remote signal survives into the saved text
        # even when a specific place name is also present. Previously
        # this used `location or ("Remote" if remote else "")`, which
        # silently DROPPED the remote flag whenever a location string was
        # present - the likely cause of on-site German postings leaking
        # into the list despite Arbeitnow's API saying remote=true/false.
        if remote_flag:
            location = f"{location} (Remote)".strip() if location else "Remote"
        job_type = ", ".join(raw_job.get("job_types", []) or [])
        location_or_type = f"{location}" + (f" | {job_type}" if job_type else "")
        description = strip_html(raw_job.get("description", ""))
        date_posted = _epoch_to_iso(raw_job.get("created_at"))

        # Arbeitnow is a general German/EU board mixing remote AND
        # on-site/hybrid roles - its explicit "remote" boolean is the
        # authoritative signal (remote_hint), not a text guess.
        remote_ok, location_or_type = _apply_remote_gate(location_or_type, description, remote_hint=remote_flag)
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("company_name", ""),
            location_or_type=location_or_type,
            description=description,
            url=raw_job.get("url", ""),
            source="arbeitnow",
            date_posted=date_posted,
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


def fetch_himalayas():
    """
    Fetch current listings from Himalayas' public jobs API
    (https://himalayas.app/jobs/api - no API key required).

    Note: "pubDate" and "expiryDate" are Unix timestamps in SECONDS, not
    ISO strings - "pubDate" is converted via _epoch_to_iso() for display,
    and "expiryDate" is passed straight through to build_normalized_job()
    as expiry_timestamp - this is the ONE source among the seven that
    exposes real expiry info, so it's the only one that can actually
    drop an expired listing (see posting_status()).

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    try:
        response = _get("https://himalayas.app/jobs/api", params={"limit": 100})
        raw_jobs = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as error:
        print(f"[himalayas] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = len(raw_jobs)
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for raw_job in raw_jobs:
        locations = raw_job.get("locationRestrictions") or []
        location_or_type = ", ".join(locations) if locations else "Remote"
        employment_type = raw_job.get("employmentType")
        if employment_type:
            location_or_type += f" | {employment_type}"
        description = strip_html(raw_job.get("description", ""))
        date_posted = _epoch_to_iso(raw_job.get("pubDate"))

        # Himalayas is a remote-jobs-only board - see REMOTE_FIRST_SOURCES.
        remote_ok, location_or_type = _apply_remote_gate(
            location_or_type, description, assume_remote_board=True
        )
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("companyName", ""),
            location_or_type=location_or_type,
            description=description,
            url=raw_job.get("applicationLink") or raw_job.get("guid") or "",
            source="himalayas",
            date_posted=date_posted,
            expiry_timestamp=raw_job.get("expiryDate"),
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


def fetch_jobicy():
    """
    Fetch current listings from Jobicy's public remote-jobs API
    (https://jobicy.com/api/v2/remote-jobs - no API key required).
    Fetched broadly (no industry/tag filter) since is_relevant() already
    does the keyword filtering, same as RemoteOK/Remotive.

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    try:
        response = _get("https://jobicy.com/api/v2/remote-jobs", params={"count": 100})
        raw_jobs = response.json().get("jobs", [])
    except (requests.RequestException, ValueError) as error:
        print(f"[jobicy] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = len(raw_jobs)
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for raw_job in raw_jobs:
        job_type = ", ".join(raw_job.get("jobType", []) or [])
        location_or_type = f"{raw_job.get('jobGeo', 'Remote')}" + (f" | {job_type}" if job_type else "")
        description = strip_html(raw_job.get("jobDescription", ""))

        # Jobicy is a remote-jobs-only board - see REMOTE_FIRST_SOURCES.
        remote_ok, location_or_type = _apply_remote_gate(
            location_or_type, description, assume_remote_board=True
        )
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=raw_job.get("jobTitle", ""),
            company=raw_job.get("companyName", ""),
            location_or_type=location_or_type,
            description=description,
            url=raw_job.get("url", ""),
            source="jobicy",
            date_posted=raw_job.get("pubDate"),
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


# The Muse's public API requires a category, so we check a couple of
# broad categories most likely to contain ML/AI/data roles - is_relevant()
# still does the real work of narrowing this down.
THE_MUSE_CATEGORIES = ["Software Engineering", "Data and Analytics"]


def fetch_themuse():
    """
    Fetch current listings from The Muse's public jobs API
    (https://www.themuse.com/api/public/jobs - no API key required),
    for each category in THE_MUSE_CATEGORIES (page 0 only, to stay
    polite - each category can have tens of thousands of results).

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    jobs = []
    found = 0
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for category in THE_MUSE_CATEGORIES:
        try:
            response = _get(
                "https://www.themuse.com/api/public/jobs", params={"category": category, "page": 0}
            )
            raw_jobs = response.json().get("results", [])
        except (requests.RequestException, ValueError) as error:
            print(f"[themuse] fetch failed for category '{category}', skipping it: {error}")
            continue

        found += len(raw_jobs)

        for raw_job in raw_jobs:
            locations = raw_job.get("locations") or []
            location_or_type = ", ".join(loc.get("name", "") for loc in locations) or "Remote"
            description = strip_html(raw_job.get("contents", ""))

            # The Muse is a general corporate job board (NOT remote-only)
            # with no explicit remote/on-site flag - real city names with
            # no remote mention (its most common case) are exactly the
            # on-site listings this filter is meant to catch, so this is
            # pure text classification, no source-level assumption.
            remote_ok, location_or_type = _apply_remote_gate(location_or_type, description)
            if not remote_ok:
                remote_filtered += 1
                continue

            job = build_normalized_job(
                title=raw_job.get("name", ""),
                company=(raw_job.get("company") or {}).get("name", ""),
                location_or_type=location_or_type,
                description=description,
                url=(raw_job.get("refs") or {}).get("landing_page", ""),
                source="themuse",
                date_posted=raw_job.get("publication_date"),
            )

            if is_wrong_domain(job["text"]):
                domain_filtered += 1
                continue

            if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
                continue
            keyword_matched += 1

            if job["freshness"] != "expired":
                jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


# WeWorkRemotely publishes one public RSS feed per category - "remote
# programming jobs" is the one most likely to carry ML/AI/data roles.
WEWORKREMOTELY_RSS_URL = "https://weworkremotely.com/categories/remote-programming-jobs.rss"

_RSS_ITEM_PATTERN = re.compile(r"<item>(.*?)</item>", re.DOTALL)


def _rss_field(item_xml, tag):
    """Pull the text of one <tag>...</tag> out of a single RSS <item> block."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", item_xml, re.DOTALL)
    return html.unescape(match.group(1).strip()) if match else ""


def fetch_weworkremotely():
    """
    Fetch current listings from WeWorkRemotely's public RSS feed
    (https://weworkremotely.com/categories/remote-programming-jobs.rss -
    no API key, no login). Parsed with plain regex/string handling
    rather than an XML parser, since RSS item bodies are simple and this
    avoids pulling in an extra XML dependency for a handful of fields.

    WWR titles are formatted "Company: Job Title" - split on the first
    ": " where possible so the company name doesn't stay stuck to the
    title.

    Returns (jobs, funnel_stats) with the same meaning as
    fetch_remoteok().
    """
    try:
        response = _get(WEWORKREMOTELY_RSS_URL)
        raw_items = _RSS_ITEM_PATTERN.findall(response.text)
    except requests.RequestException as error:
        print(f"[weworkremotely] fetch failed, skipping this source: {error}")
        return [], _empty_funnel()

    jobs = []
    found = len(raw_items)
    keyword_matched = 0
    remote_filtered = 0
    domain_filtered = 0

    for item_xml in raw_items:
        raw_title = _rss_field(item_xml, "title")
        if ": " in raw_title:
            company, title = raw_title.split(": ", 1)
        else:
            company, title = "", raw_title

        region = _rss_field(item_xml, "region")
        category = _rss_field(item_xml, "category")
        location_or_type = f"{region or 'Remote'}" + (f" | {category}" if category else "")

        description_html = _rss_field(item_xml, "description")
        description = strip_html(description_html)
        url = _rss_field(item_xml, "link")
        date_posted = _rfc2822_to_iso(_rss_field(item_xml, "pubDate"))

        # WeWorkRemotely is a remote-jobs-only board - see REMOTE_FIRST_SOURCES.
        remote_ok, location_or_type = _apply_remote_gate(
            location_or_type, description, assume_remote_board=True
        )
        if not remote_ok:
            remote_filtered += 1
            continue

        job = build_normalized_job(
            title=title,
            company=company,
            location_or_type=location_or_type,
            description=description,
            url=url,
            source="weworkremotely",
            date_posted=date_posted,
        )

        if is_wrong_domain(job["text"]):
            domain_filtered += 1
            continue

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "expired":
            jobs.append(job)

    return jobs, {
        "found": found,
        "keyword_matched": keyword_matched,
        "remote_filtered": remote_filtered,
        "domain_filtered": domain_filtered,
    }


# One entry per source: name -> fetch function. To add a new source,
# write a fetch_<name>() following the same (jobs, funnel_stats) contract
# as the ones above, and add it here - fetch_all_jobs() and the --fetch
# CLI summary pick it up automatically.
SOURCE_FETCHERS = {
    "remoteok": fetch_remoteok,
    "remotive": fetch_remotive,
    "arbeitnow": fetch_arbeitnow,
    "himalayas": fetch_himalayas,
    "jobicy": fetch_jobicy,
    "themuse": fetch_themuse,
    "weworkremotely": fetch_weworkremotely,
}


def fetch_all_jobs(existing_jobs):
    """
    Call every source in SOURCE_FETCHERS, dedup the results against
    `existing_jobs` (the full list of job dicts already in jobs.json -
    see match_llm.py's cli_fetch_jobs()) AND against each other (the
    same real-world job can appear on multiple boards), and return
    (new_jobs, stats).

    Two dedup keys are checked, either one is enough to reject a job:
    - the EXACT text hash (hash_job_text()) - unchanged from before.
    - the NORMALIZED title+company key (normalize_company_title_key()) -
      catches the same real posting re-served with a slightly different
      description snapshot (whitespace/formatting) that would otherwise
      produce a different exact hash and slip through as a "new" job.

    `stats` is a dict keyed by source name:
        {"remoteok": {"found": N, "keyword_matched": K, "remote_filtered": R,
        "domain_filtered": W, "active": M, "duplicates": D, "added": A}, ...}
    - "found": total real listings the source returned.
    - "remote_filtered": how many were dropped for being on-site/hybrid
      (STEP 2's remote-only gate - see classify_remote_status()).
    - "domain_filtered": how many were dropped for being the wrong field
      or a non-technical role (STEP 4's negative-keyword gate - see
      is_wrong_domain()).
    - "keyword_matched": how many of those matched an ML/AI keyword
      (checked AFTER the two filters above, so this reflects the
      relevant-AND-remote-AND-right-domain count).
    - "active": how many of the keyword-matched jobs are still an active
      listing (i.e. NOT dropped as expired - see posting_status()) - this
      is what actually gets considered for jobs.json. For every source
      except Himalayas this always equals "keyword_matched", since only
      Himalayas exposes real expiry info to drop postings by.
    - "duplicates": how many of those were already in jobs.json (by
      either dedup key) OR already added by an earlier source in this
      same run (cross-source dedup - the same posting on two boards only
      gets added once).
    - "added": active AND relevant AND new (== len of what actually gets
      appended).

    A single source raising an unexpected exception (network hiccup, API
    shape change, etc.) is caught here too, on top of each fetch_*()'s
    own try/except, so one bad source can never take down the whole run -
    it just contributes zero jobs and gets reported as all-zero stats.
    """
    new_jobs = []
    stats = {}
    seen_hashes_this_run = {hash_job_text(job.get("text", "")) for job in existing_jobs}
    seen_keys_this_run = {normalize_company_title_key(job.get("text", "")) for job in existing_jobs}
    seen_keys_this_run.discard(None)

    for source_name, fetch_function in SOURCE_FETCHERS.items():
        try:
            active_relevant_jobs, funnel_stats = fetch_function()
        except Exception as error:  # noqa: BLE001 - one bad source must not crash --fetch/--daily
            print(f"[{source_name}] unexpected error, skipping this source: {error}")
            active_relevant_jobs, funnel_stats = [], _empty_funnel()

        duplicates = 0
        added = 0
        for job in active_relevant_jobs:
            job_hash = hash_job_text(job["text"])
            job_key = normalize_company_title_key(job["text"])
            if job_hash in seen_hashes_this_run or (job_key is not None and job_key in seen_keys_this_run):
                duplicates += 1
                continue
            seen_hashes_this_run.add(job_hash)
            if job_key is not None:
                seen_keys_this_run.add(job_key)
            new_jobs.append(job)
            added += 1

        stats[source_name] = {
            "found": funnel_stats["found"],
            "remote_filtered": funnel_stats.get("remote_filtered", 0),
            "domain_filtered": funnel_stats.get("domain_filtered", 0),
            "keyword_matched": funnel_stats["keyword_matched"],
            "active": len(active_relevant_jobs),
            "duplicates": duplicates,
            "added": added,
        }

    return new_jobs, stats
