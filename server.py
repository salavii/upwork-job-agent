"""
server.py

Part 1 of the semi-automatic Upwork job grabber.

What this script does, and ONLY this:
- Runs a small local HTTP server at http://localhost:8765.
- Accepts POST requests containing job text.
- Appends that job text to jobs.txt, followed by a "---" separator line
  (the same format match_llm.py already expects when it reads jobs.txt).
- Allows requests from a browser (CORS enabled), so a future browser
  extension (Part 2, not built yet) can call this server directly from
  an Upwork job page.
- Prints a short confirmation line in the terminal every time a job is
  saved.

Only the Python standard library is used (http.server), so there is
nothing extra to install - just run this file with Python.

How the browser extension (later) is expected to call this:
    POST http://localhost:8765/save_job
    Body: either
      - raw text: the job description as plain text, OR
      - JSON: {"job_text": "the job description..."}
Both forms are accepted (see handle_save_job() below for details).
"""

# http.server is the standard library's basic HTTP server toolkit.
# - HTTPServer: the actual server that listens on a port and accepts
#   connections.
# - BaseHTTPRequestHandler: a class we extend to define what happens
#   when a request comes in (we only care about POST requests here).
from http.server import BaseHTTPRequestHandler, HTTPServer

# `json` lets us try to parse the request body as JSON (in case the
# caller sends {"job_text": "..."} instead of raw text).
import json

# The local address and port this server listens on. "localhost" means
# "only accept connections from this same machine" - fine for a personal
# tool like this.
SERVER_HOST = "localhost"
SERVER_PORT = 8765

# The same jobs file that match_llm.py already reads from, so any job
# saved here shows up next time match_llm.py is run.
JOBS_FILE = "jobs.txt"

# The separator line match_llm.py's load_jobs() splits on. Must match
# exactly what match_llm.py expects.
JOB_SEPARATOR = "---"


def append_job_to_file(job_text):
    """
    Add `job_text` to the end of JOBS_FILE, followed by a separator line,
    so match_llm.py's load_jobs() can split the file back into separate
    jobs later.

    We open the file in "a" (append) mode, which creates the file if it
    doesn't exist yet, and otherwise adds to the end without erasing
    anything already there.
    """
    with open(JOBS_FILE, "a", encoding="utf-8") as f:
        # A leading newline keeps this job visually separated from
        # whatever was written before it, even if the previous entry
        # didn't end with a trailing newline.
        f.write(f"\n{job_text.strip()}\n{JOB_SEPARATOR}\n")


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

    def do_POST(self):
        """
        Handle an incoming POST request: read the job text from the
        request body, save it to jobs.txt, and reply with a small JSON
        confirmation.
        """
        # "Content-Length" tells us how many bytes of body to read.
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length).decode("utf-8")

        # The body might be raw text, or JSON like {"job_text": "..."}.
        # Try JSON first; if that fails, just treat the whole body as
        # the job text itself.
        job_text = raw_body
        try:
            parsed_body = json.loads(raw_body)
            if isinstance(parsed_body, dict) and "job_text" in parsed_body:
                job_text = parsed_body["job_text"]
        except json.JSONDecodeError:
            pass  # raw_body wasn't JSON - that's fine, use it as-is

        job_text = job_text.strip()

        if not job_text:
            # Nothing usable was sent - reply with an error instead of
            # writing an empty entry to jobs.txt.
            self.send_response(400)  # 400 = "Bad Request"
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No job text received"}).encode("utf-8"))
            return

        append_job_to_file(job_text)

        # Print a confirmation in the terminal so it's obvious a job was
        # saved, and roughly how big it was.
        print(f"Saved job ({len(job_text)} chars) to {JOBS_FILE}")

        self.send_response(200)  # 200 = "OK"
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "saved", "chars": len(job_text)}).encode("utf-8"))

    def log_message(self, format, *args):
        """
        BaseHTTPRequestHandler normally prints a raw HTTP log line for
        every request (method, path, status code). We override it with
        nothing (`pass`) to keep the terminal output limited to just our
        own "Saved job (...) to jobs.txt" confirmation messages.
        """
        pass


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python server.py`), not
# when it's imported by another file."
if __name__ == "__main__":
    server = HTTPServer((SERVER_HOST, SERVER_PORT), JobGrabberRequestHandler)

    print(f"Job grabber server running at http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"POST job text to http://{SERVER_HOST}:{SERVER_PORT}/save_job")
    print("Press Ctrl+C to stop.\n")

    try:
        # serve_forever() blocks here, handling one request at a time,
        # until the process is interrupted (e.g. Ctrl+C).
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()
