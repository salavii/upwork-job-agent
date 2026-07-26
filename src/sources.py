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
        "freshness": "fresh" | "unknown",
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

# Only jobs posted within this many days are kept - the whole point of
# --daily is fresh, low-competition postings, not a backlog of listings
# from weeks ago that are already buried under other applicants. A job
# whose source doesn't expose a usable date at all is kept anyway (with
# "freshness": "unknown") rather than dropped - see freshness_status().
MAX_JOB_AGE_DAYS = 2

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
    r"deep learning",
    r"\bAI\b",
    r"\bLLM\b",
    r"large language model",
    r"\bNLP\b",
    r"natural language processing",
    r"\bRAG\b",
    r"retrieval[- ]augmented generation",
    r"computer vision",
    r"PyTorch",
    r"TensorFlow",
    r"fine[- ]?tuning",
    r"prompt engineering",
    r"hugging face",
    r"data scientist",
    r"model training",
    r"embeddings?",
    r"vector database",
]
RELEVANT_KEYWORDS_PATTERN = re.compile("|".join(KEYWORD_PATTERNS), re.IGNORECASE)

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


def freshness_status(date_str):
    """
    Classify a job's posted-date (already normalized to an ISO string by
    the caller) as:
      - "fresh":   posted within MAX_JOB_AGE_DAYS - keep it.
      - "stale":   older than that - drop it, that's the whole point of
        the freshness filter (fresh, low-competition postings only).
      - "unknown": missing or unparseable - the source just doesn't
        expose a reliable date. Kept (we'd rather show an unverified-date
        job than silently lose a real match), but tagged so the report
        never confuses it with a provably-fresh one.
    """
    posted_at = _parse_iso_datetime(date_str)
    if posted_at is None:
        return "unknown"

    age = datetime.datetime.now(datetime.timezone.utc) - posted_at
    return "fresh" if age <= datetime.timedelta(days=MAX_JOB_AGE_DAYS) else "stale"


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


def build_normalized_job(*, title, company, location_or_type, description, url, source, date_posted=None):
    """
    Build one job dict in the exact shape jobs.json expects (see the
    module docstring above). "text" is built as
    title + company + location/type + full description, matching what
    match_llm.py's extract_title_and_metadata() expects (title = first
    line). `date_posted` must already be an ISO-ish string (or None) -
    each fetch_*() converts its source's native date format before
    calling this.
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
        "freshness": freshness_status(date_posted),
    }


def _empty_funnel():
    return {"found": 0, "keyword_matched": 0}


def fetch_remoteok():
    """
    Fetch current listings from RemoteOK's public JSON API
    (https://remoteok.com/api - no API key required).

    The API returns a JSON array whose FIRST element is a legal/notice
    blob (no "position"/"company" keys), not a real job - skipped below.

    Returns (jobs, funnel_stats): `jobs` is the list of normalized jobs
    that passed BOTH the keyword-relevance filter and the freshness
    filter (fresh or unknown-date, never stale). `funnel_stats` is
    {"found": N, "keyword_matched": M} - N is how many real listings the
    API returned in total, M is how many matched a keyword (regardless of
    age) - used for the --fetch summary. Returns ([], funnel) with zeroed
    counts if the request fails for any reason - one source failing must
    never crash the run.
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

    for raw_job in raw_jobs:
        if "position" not in raw_job or "company" not in raw_job:
            continue  # the legal-notice entry, or a malformed row

        found += 1

        location = raw_job.get("location") or "Remote"
        job_type = ", ".join(raw_job.get("tags", []) or [])
        location_or_type = f"{location}" + (f" | {job_type}" if job_type else "")

        job = build_normalized_job(
            title=raw_job.get("position", ""),
            company=raw_job.get("company", ""),
            location_or_type=location_or_type,
            description=strip_html(raw_job.get("description", "")),
            url=raw_job.get("url") or raw_job.get("apply_url") or "",
            source="remoteok",
            date_posted=raw_job.get("date"),
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


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

    for raw_job in raw_jobs:
        location_or_type = f"{raw_job.get('candidate_required_location', 'Remote')} | {raw_job.get('job_type', '')}".strip(
            " |"
        )

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("company_name", ""),
            location_or_type=location_or_type,
            description=strip_html(raw_job.get("description", "")),
            url=raw_job.get("url", ""),
            source="remotive",
            date_posted=raw_job.get("publication_date"),
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


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

    for raw_job in raw_jobs:
        location = raw_job.get("location") or ("Remote" if raw_job.get("remote") else "")
        job_type = ", ".join(raw_job.get("job_types", []) or [])
        location_or_type = f"{location}" + (f" | {job_type}" if job_type else "")
        date_posted = _epoch_to_iso(raw_job.get("created_at"))

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("company_name", ""),
            location_or_type=location_or_type,
            description=strip_html(raw_job.get("description", "")),
            url=raw_job.get("url", ""),
            source="arbeitnow",
            date_posted=date_posted,
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


def fetch_himalayas():
    """
    Fetch current listings from Himalayas' public jobs API
    (https://himalayas.app/jobs/api - no API key required).

    Note: "pubDate" is a Unix timestamp in SECONDS, not an ISO string -
    converted via _epoch_to_iso() before freshness/date handling.

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

    for raw_job in raw_jobs:
        locations = raw_job.get("locationRestrictions") or []
        location_or_type = ", ".join(locations) if locations else "Remote"
        employment_type = raw_job.get("employmentType")
        if employment_type:
            location_or_type += f" | {employment_type}"
        date_posted = _epoch_to_iso(raw_job.get("pubDate"))

        job = build_normalized_job(
            title=raw_job.get("title", ""),
            company=raw_job.get("companyName", ""),
            location_or_type=location_or_type,
            description=strip_html(raw_job.get("description", "")),
            url=raw_job.get("applicationLink") or raw_job.get("guid") or "",
            source="himalayas",
            date_posted=date_posted,
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


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

    for raw_job in raw_jobs:
        job_type = ", ".join(raw_job.get("jobType", []) or [])
        location_or_type = f"{raw_job.get('jobGeo', 'Remote')}" + (f" | {job_type}" if job_type else "")

        job = build_normalized_job(
            title=raw_job.get("jobTitle", ""),
            company=raw_job.get("companyName", ""),
            location_or_type=location_or_type,
            description=strip_html(raw_job.get("jobDescription", "")),
            url=raw_job.get("url", ""),
            source="jobicy",
            date_posted=raw_job.get("pubDate"),
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


# The Muse's public API requires a category and doesn't reliably sort by
# recency, so we check a couple of broad categories most likely to
# contain ML/AI/data roles - is_relevant() + the freshness filter still
# do the real work of narrowing this down.
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

            job = build_normalized_job(
                title=raw_job.get("name", ""),
                company=(raw_job.get("company") or {}).get("name", ""),
                location_or_type=location_or_type,
                description=strip_html(raw_job.get("contents", "")),
                url=(raw_job.get("refs") or {}).get("landing_page", ""),
                source="themuse",
                date_posted=raw_job.get("publication_date"),
            )

            if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
                continue
            keyword_matched += 1

            if job["freshness"] != "stale":
                jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


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
        url = _rss_field(item_xml, "link")
        date_posted = _rfc2822_to_iso(_rss_field(item_xml, "pubDate"))

        job = build_normalized_job(
            title=title,
            company=company,
            location_or_type=location_or_type,
            description=strip_html(description_html),
            url=url,
            source="weworkremotely",
            date_posted=date_posted,
        )

        if not is_relevant(job["text"]) or is_dead_posting(job["text"]):
            continue
        keyword_matched += 1

        if job["freshness"] != "stale":
            jobs.append(job)

    return jobs, {"found": found, "keyword_matched": keyword_matched}


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
        {"remoteok": {"found": N, "keyword_matched": K, "fresh": M, "duplicates": D, "added": A}, ...}
    - "found": total real listings the source returned.
    - "keyword_matched": how many of those matched an ML/AI keyword,
      regardless of age.
    - "fresh": how many of the keyword-matched jobs were ALSO fresh or
      unknown-date (i.e. NOT filtered out as stale) - this is what
      actually gets considered for jobs.json.
    - "duplicates": how many of those were already in jobs.json (by
      either dedup key) OR already added by an earlier source in this
      same run (cross-source dedup - the same posting on two boards only
      gets added once).
    - "added": fresh AND relevant AND new (== len of what actually gets
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
            fresh_relevant_jobs, funnel_stats = fetch_function()
        except Exception as error:  # noqa: BLE001 - one bad source must not crash --fetch/--daily
            print(f"[{source_name}] unexpected error, skipping this source: {error}")
            fresh_relevant_jobs, funnel_stats = [], _empty_funnel()

        duplicates = 0
        added = 0
        for job in fresh_relevant_jobs:
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
            "keyword_matched": funnel_stats["keyword_matched"],
            "fresh": len(fresh_relevant_jobs),
            "duplicates": duplicates,
            "added": added,
        }

    return new_jobs, stats
