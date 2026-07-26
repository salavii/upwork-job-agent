"""
server.py

Part 1 of the semi-automatic Upwork job grabber.

What this script does, and ONLY this:
- Runs a small local HTTP server at http://localhost:8765.
- Accepts POST requests containing job text.
- Stores each job as a structured entry in jobs.json (a JSON list),
  including when it was found - not just as raw text anymore.
- Skips saving a job if a very similar one has already been saved
  (deduplication), so re-visiting or re-clicking on the same posting
  doesn't pile up duplicate entries.
- Allows requests from a browser (CORS enabled), so the browser
  extension (Part 2) can call this server directly from an Upwork job
  page.
- Prints a short confirmation line in the terminal every time a job is
  saved OR skipped as a duplicate.

Only the Python standard library is used (http.server, json, difflib,
datetime), so there is nothing extra to install - just run this file
with Python.

Note on jobs.txt: earlier versions of this server appended plain text to
jobs.txt. That file (if it exists) is left alone and untouched - we just
don't write to it anymore. Going forward, everything lives in jobs.json.

Note on match_llm.py: after a successful delete, this server calls
match_llm.build_report() so report.html is rebuilt immediately with the
deleted job gone - see _handle_delete_job() below. That function only
reads jobs.json/results.json and writes report.html; it never calls
Ollama, so deleting a job here doesn't trigger any (re-)scoring.

How the browser extension calls this:
    POST http://localhost:8765/save_job
    Body: either
      - raw text: the job description as plain text, OR
      - JSON: {"job_text": "the job description..."}
Both forms are accepted (see do_POST() below for details).
"""

# http.server is the standard library's basic HTTP server toolkit.
# - HTTPServer: the actual server that listens on a port and accepts
#   connections.
# - BaseHTTPRequestHandler: a class we extend to define what happens
#   when a request comes in (we only care about POST requests here).
from http.server import BaseHTTPRequestHandler, HTTPServer

# `json` is used both to parse incoming request bodies and to read/write
# jobs.json.
import json

# `datetime` gives us today's date and a full timestamp for each saved job.
import datetime

# `difflib` (standard library) is used for text-similarity comparisons,
# so we can catch near-duplicate job postings, not just exact ones.
import difflib

# `os` is used just to check whether jobs.json already exists.
import os

# `hashlib` is used to identify a job by a hash of its text - the same
# scheme match_llm.py uses for its results.json cache - so the HTML
# report's delete button can tell this server exactly which job to
# remove without needing to send the entire job text back.
import hashlib

# We reuse match_llm.py's build_report() so report.html can be rebuilt
# right after a delete, instead of duplicating its HTML-building logic
# here. Importing match_llm.py does NOT call Ollama or do any scoring by
# itself (that only happens inside its own `if __name__ == "__main__":`
# block) - build_report() only reads jobs.json/results.json and writes
# report.html from whatever is already cached.
import match_llm

# The local address and port this server listens on. "localhost" means
# "only accept connections from this same machine" - fine for a personal
# tool like this.
SERVER_HOST = "localhost"
SERVER_PORT = 8765

# Where jobs are now stored: a JSON list of job objects, each shaped like
#   {"text": "...", "date_added": "YYYY-MM-DD", "saved_at": "2026-07-17T14:32:05"}
JOBS_JSON_FILE = "jobs.json"

# Where match_llm.py caches each job's scored result, keyed by the same
# job-text hash scheme as JOBS_JSON_FILE. When a job is deleted here, its
# cached result (if any) is deleted too, so it doesn't linger unused.
RESULTS_JSON_FILE = "results.json"

# Two jobs are considered duplicates if either:
# - their first line (title) matches, ignoring case/whitespace, OR
# - their full text is at least this similar (0.0 = nothing alike,
#   1.0 = identical). 0.9 catches things like copy-pasting the same job
#   with a tiny formatting difference, without flagging two genuinely
#   different jobs that just happen to share some boilerplate wording.
DUPLICATE_SIMILARITY_THRESHOLD = 0.9


def load_jobs():
    """
    Read jobs.json and return its contents as a Python list of job
    dicts. If the file doesn't exist yet (first run) or is somehow not
    valid JSON, we return an empty list instead of crashing - the very
    next save will create/rewrite the file correctly.
    """
    if not os.path.exists(JOBS_JSON_FILE):
        return []

    with open(JOBS_JSON_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_jobs(jobs):
    """
    Write the full `jobs` list back to jobs.json, formatted for easy
    reading (indent=2). We always rewrite the whole file rather than
    appending, since JSON isn't append-friendly (it's one big list, not
    line-by-line like jobs.txt was).
    """
    with open(JOBS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def load_results():
    """
    Read RESULTS_JSON_FILE (match_llm.py's score/proposal cache) and
    return it as a dict. Returns an empty dict if the file doesn't exist
    or isn't valid JSON - deleting a job just becomes a no-op on the
    cache side in that case, which is fine.
    """
    if not os.path.exists(RESULTS_JSON_FILE):
        return {}

    with open(RESULTS_JSON_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_results(results):
    """
    Write the `results` cache dict back to RESULTS_JSON_FILE.
    """
    with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def job_text_hash(job_text):
    """
    Turn a job's text into a short, stable identifier (a SHA-256 hash).
    This MUST use the exact same formula as match_llm.py's
    hash_job_text() (strip the text, encode as UTF-8, SHA-256, hex
    digest), since the HTML report generated by match_llm.py embeds
    this hash as each job's id, and the delete button sends that same
    id back here.
    """
    return hashlib.sha256(job_text.strip().encode("utf-8")).hexdigest()


def get_title(job_text):
    """
    Return a job's first line, cleaned up for comparison purposes
    (trimmed and lowercased), which we treat as its "title" for
    deduplication.
    """
    first_line = job_text.strip().splitlines()[0] if job_text.strip() else ""
    return first_line.strip().lower()


def is_duplicate(new_job_text, existing_jobs):
    """
    Check whether `new_job_text` is a duplicate of any job already in
    `existing_jobs`.

    Two checks are used together:
    1. Same title (first line) - catches the common case of saving the
       exact same posting twice (e.g. clicking the extension button
       twice on the same page).
    2. High overall text similarity (via difflib.SequenceMatcher) -
       catches near-duplicates where the title might differ slightly
       (extra whitespace, a re-scrape with minor formatting changes) but
       the body is essentially the same job.
    """
    new_title = get_title(new_job_text)

    for existing_job in existing_jobs:
        existing_text = existing_job.get("text", "")

        if new_title and new_title == get_title(existing_text):
            return True

        similarity = difflib.SequenceMatcher(None, new_job_text, existing_text).ratio()
        if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
            return True

    return False


class JobGrabberRequestHandler(BaseHTTPRequestHandler):
    """
    Defines how this server responds to incoming HTTP requests.
    BaseHTTPRequestHandler calls do_POST() automatically whenever a POST
    request comes in, and do_OPTIONS() for the CORS "preflight" request
    browsers send before certain POST requests.
    """

    def _send_cors_headers(self):
        """
        Send the headers that tell a browser "it's OK for a web page to
        call this server from JavaScript, even though the web page and
        this server are on different origins."

        Without this, a browser extension's content script (running on
        an upwork.com page) would be blocked by the browser itself from
        calling http://localhost:8765, due to the browser's CORS policy.
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """
        Handle CORS "preflight" requests. Before a browser sends certain
        POST requests to a different origin, it first sends an OPTIONS
        request to ask "am I allowed to do this?". We just need to
        answer "yes" with the CORS headers and an empty body.
        """
        self.send_response(204)  # 204 = "No Content", a normal empty OK response
        self._send_cors_headers()
        self.end_headers()

    def _send_json_response(self, status_code, payload):
        """
        Small helper to send a JSON response with the right headers,
        used by every branch of do_POST() below.
        """
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        """
        Route an incoming POST request to the right handler based on
        its path: /save_job (from the browser extension), /delete_job
        (from report.html's delete buttons), /remove_daily_job and
        /clear_daily_list (from daily_report.html's persistent job list -
        see match_llm.py's update_persistent_job_list()). Anything else
        gets a 404.
        """
        if self.path == "/save_job":
            self._handle_save_job()
        elif self.path == "/delete_job":
            self._handle_delete_job()
        elif self.path == "/remove_daily_job":
            self._handle_remove_daily_job()
        elif self.path == "/clear_daily_list":
            self._handle_clear_daily_list()
        else:
            self._send_json_response(404, {"error": f"Unknown endpoint: {self.path}"})

    def _read_json_body(self):
        """
        Read the request body and try to parse it as JSON. Returns
        (parsed_dict, raw_body, was_valid_json_dict). The third value
        matters: it lets callers tell "this wasn't JSON at all, treat it
        as plain text" apart from "this WAS valid JSON, just not the
        shape we expected" - conflating those two used to let a
        wrong-shaped JSON body (e.g. a /delete_job-style
        {"job_id": "..."} body sent to /save_job) get saved verbatim as
        if it were a plain-text job description.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            parsed = json.loads(raw_body)
            is_dict = isinstance(parsed, dict)
            return (parsed if is_dict else {}), raw_body, is_dict
        except json.JSONDecodeError:
            return {}, raw_body, False

    def _handle_save_job(self):
        """
        Read the job text from the request body, check it against
        existing jobs for duplicates, save it to jobs.json if it's new,
        and reply with a small JSON confirmation either way.
        """
        parsed_body, raw_body, was_json_dict = self._read_json_body()

        # The body might be raw text, or JSON like {"job_text": "..."}.
        if was_json_dict and "job_text" in parsed_body:
            job_text = parsed_body["job_text"]
        elif not was_json_dict:
            # Not JSON at all - treat the whole raw body as plain-text
            # job description.
            job_text = raw_body
        else:
            # It WAS valid JSON, just not shaped like a save_job request
            # (e.g. a delete_job-style {"job_id": ...} body sent here by
            # mistake). Don't save that JSON verbatim as a "job".
            job_text = ""

        job_text = job_text.strip()

        if not job_text:
            # Nothing usable was sent - reply with an error instead of
            # writing an empty (or wrong-shaped) entry to jobs.json.
            self._send_json_response(400, {"error": "No job text received"})
            return

        existing_jobs = load_jobs()

        if is_duplicate(job_text, existing_jobs):
            print(f"Duplicate job skipped ({len(job_text)} chars) - already in {JOBS_JSON_FILE}")
            self._send_json_response(
                200, {"status": "duplicate", "message": "duplicate, skipped"}
            )
            return

        now = datetime.datetime.now()
        new_job = {
            "text": job_text,
            "date_added": now.strftime("%Y-%m-%d"),
            "saved_at": now.isoformat(timespec="seconds"),
        }
        existing_jobs.append(new_job)
        save_jobs(existing_jobs)

        # Print a confirmation in the terminal so it's obvious a job was
        # saved, and roughly how big it was.
        print(f"Saved job ({len(job_text)} chars) to {JOBS_JSON_FILE}")

        self._send_json_response(200, {"status": "saved", "chars": len(job_text)})

    def _handle_delete_job(self):
        """
        Remove a job from jobs.json, identified either by "job_id" (a
        job-text hash, as embedded in report.html by match_llm.py) or by
        "title" (its first line) as a fallback. Also removes that job's
        cached result from results.json if one exists.
        """
        parsed_body, _, _ = self._read_json_body()
        job_id = parsed_body.get("job_id")
        title_query = parsed_body.get("title")

        if not job_id and not title_query:
            self._send_json_response(400, {"error": "No job_id or title provided"})
            return

        jobs = load_jobs()
        match_index = None

        for index, job in enumerate(jobs):
            job_text = job.get("text", "")
            if job_id and job_text_hash(job_text) == job_id:
                match_index = index
                break
            if title_query and get_title(job_text) == title_query.strip().lower():
                match_index = index
                break

        if match_index is None:
            self._send_json_response(404, {"error": "Job not found"})
            return

        removed_job = jobs.pop(match_index)
        save_jobs(jobs)

        # Also remove the job's cached score/proposal, if it has one, so
        # a deleted job doesn't linger unused in results.json.
        removed_text = removed_job.get("text", "")
        results = load_results()
        removed_hash = job_text_hash(removed_text)
        had_cached_result = removed_hash in results
        if had_cached_result:
            del results[removed_hash]
            save_results(results)

        title = get_title(removed_text)
        cache_note = " (also removed its cached result)" if had_cached_result else ""
        print(f"Deleted job: {title}{cache_note}")

        # Rebuild report.html right away so it no longer shows the
        # deleted job. This only reads jobs.json/results.json and writes
        # report.html - it never calls Ollama, so this doesn't trigger
        # any (re-)scoring.
        match_llm.build_report()
        print(f"Rebuilt {match_llm.REPORT_FILE}")

        self._send_json_response(200, {"status": "deleted", "title": title})

    def _handle_remove_daily_job(self):
        """
        Permanently remove one job from match_llm.py's PERSISTENT job
        list (job_list.json - see update_persistent_job_list()), called
        by daily_report.html's per-card "Remove" (x) button. The job's
        hash is recorded in "removed_hashes" too, so a later --fetch/
        --daily re-discovering the same posting can never silently
        re-add it - this is a real deletion, not just a display filter.
        """
        parsed_body, _, _ = self._read_json_body()
        job_id = parsed_body.get("job_id")

        if not job_id:
            self._send_json_response(400, {"error": "No job_id provided"})
            return

        job_list = match_llm.load_job_list()
        removed_entry = job_list["jobs"].pop(job_id, None)

        if removed_entry is None:
            self._send_json_response(404, {"error": "Job not found in list"})
            return

        if job_id not in job_list["removed_hashes"]:
            job_list["removed_hashes"].append(job_id)
        match_llm.save_job_list(job_list)

        print(f"Removed from job list: {removed_entry['title']}")

        match_llm.rebuild_daily_report()
        print(f"Rebuilt {match_llm.DAILY_REPORT_FILE}")

        self._send_json_response(200, {"status": "removed", "title": removed_entry["title"]})

    def _handle_clear_daily_list(self):
        """
        Wipe match_llm.py's PERSISTENT job list entirely (both the
        current jobs AND the removed-hashes tombstone list, since this
        is meant to be a genuine fresh start - see daily_report.html's
        "Clear all" button), then rebuild the now-empty daily_report.html.
        """
        match_llm.save_job_list({"jobs": {}, "removed_hashes": []})
        print("Cleared the entire persistent job list.")

        match_llm.rebuild_daily_report()
        print(f"Rebuilt {match_llm.DAILY_REPORT_FILE}")

        self._send_json_response(200, {"status": "cleared"})

    def log_message(self, format, *args):
        """
        BaseHTTPRequestHandler normally prints a raw HTTP log line for
        every request (method, path, status code). We override it with
        nothing (`pass`) to keep the terminal output limited to just our
        own "Saved job (...)" / "Duplicate job skipped (...)" messages.
        """
        pass


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python server.py`), not
# when it's imported by another file."
if __name__ == "__main__":
    server = HTTPServer((SERVER_HOST, SERVER_PORT), JobGrabberRequestHandler)

    print(f"Job grabber server running at http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"POST job text to http://{SERVER_HOST}:{SERVER_PORT}/save_job")
    print(f"POST {{\"job_id\": ...}} to http://{SERVER_HOST}:{SERVER_PORT}/delete_job to delete a job")
    print("Press Ctrl+C to stop.\n")

    try:
        # serve_forever() blocks here, handling one request at a time,
        # until the process is interrupted (e.g. Ctrl+C).
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
