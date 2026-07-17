// popup.js
//
// Runs inside the extension's popup window when it's open. Handles the
// "Save this job" button click: injects content.js into the current
// tab to read the job posting, then POSTs that text to the local
// server (server.py, from Part 1) so it gets appended to jobs.txt.

const SERVER_URL = "http://localhost:8765/save_job";

const saveButton = document.getElementById("save-button");
const statusEl = document.getElementById("status");

function showStatus(message, kind) {
  // `kind` is "success" or "error" - just controls the text color via
  // the matching CSS class defined in popup.html.
  statusEl.textContent = message;
  statusEl.className = kind;
}

saveButton.addEventListener("click", async () => {
  saveButton.disabled = true;
  showStatus("Reading job from page...", "");

  try {
    // Find the tab the user currently has focused, so we know where to
    // inject content.js.
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!activeTab || !activeTab.id) {
      showStatus("Could not find the active tab.", "error");
      return;
    }

    // Run content.js inside the page. Its final expression's return
    // value comes back as `results[0].result`.
    const injectionResults = await chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      files: ["content.js"],
    });

    const jobText = injectionResults && injectionResults[0] && injectionResults[0].result;

    if (!jobText || jobText.trim().length === 0) {
      showStatus("Couldn't find any job text on this page.", "error");
      return;
    }

    showStatus("Saving job...", "");

    // Send the extracted text to our local server (server.py). If the
    // server isn't running, this fetch will throw, which we catch below
    // and report as a clear error instead of a silent failure.
    const response = await fetch(SERVER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_text: jobText }),
    });

    if (!response.ok) {
      showStatus(`Server responded with an error (${response.status}).`, "error");
      return;
    }

    showStatus("Job saved!", "success");
  } catch (error) {
    // This is the common failure mode: the local server (server.py)
    // isn't running, so fetch() couldn't connect at all.
    showStatus("Couldn't reach the local server. Is server.py running?", "error");
  } finally {
    saveButton.disabled = false;
  }
});
