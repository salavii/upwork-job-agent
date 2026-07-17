// content.js
//
// This file is injected into the current tab (on demand, when the popup
// button is clicked - see popup.js) to read the Upwork job posting on
// the page and return it as a single cleaned-up text string.
//
// Upwork's page structure/CSS class names can change over time, so this
// is written DEFENSIVELY: it tries a few likely selectors/strategies for
// each part of the job (title, metadata, description, skills), but if
// none of them match, it falls back to grabbing the page's main visible
// text instead of failing outright. Some noise in the fallback case is
// an acceptable tradeoff for "never come back with nothing."

function extractUpworkJobText() {
  // Small helper: given a list of CSS selectors, return the trimmed
  // innerText of the first one that actually matches something on the
  // page (and where that something has non-empty text). This is what
  // makes the extraction "defensive" - if Upwork changes their markup
  // and one selector stops working, we just move on to the next guess.
  function textFromFirstMatch(selectors) {
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element && element.innerText && element.innerText.trim().length > 0) {
        return element.innerText.trim();
      }
    }
    return null;
  }

  // --- Title ---
  // Small helper used only for the title's last-resort fallback: find
  // the first heading of `tag` that is actually the job title, not a
  // hidden accessibility landmark heading (like a visually-hidden
  // "Footer navigation" <h2> screen readers use to jump around the
  // page) and not a heading that lives inside <nav>/<footer>/<header>.
  function firstVisibleHeadingOutsideChrome(tag) {
    const candidates = Array.from(document.querySelectorAll(tag));
    for (const el of candidates) {
      if (el.closest("nav, footer, header")) continue; // skip site chrome
      if (!el.offsetParent) continue; // skip visually-hidden elements
      const text = el.innerText && el.innerText.trim();
      if (text) return text;
    }
    return null;
  }

  // Helper to strip a trailing " - Upwork" (or similar) site suffix off
  // a title-like string.
  function stripSiteSuffix(text) {
    return text.replace(/\s*[-|]\s*Upwork.*$/i, "").trim();
  }

  // Prefer document.title and og:title FIRST - on Upwork these reliably
  // hold the real job title, and unlike querying the page's headings,
  // they can't accidentally match unrelated nav/footer landmarks (which
  // is what happened when this used to try <h1>/<h2> first and picked
  // up a hidden "Footer navigation" heading instead of the job title).
  let title = null;

  if (document.title && document.title.trim().length > 0) {
    title = stripSiteSuffix(document.title);
  }

  if (!title) {
    const ogTitleMeta = document.querySelector('meta[property="og:title"]');
    if (ogTitleMeta && ogTitleMeta.content && ogTitleMeta.content.trim().length > 0) {
      title = stripSiteSuffix(ogTitleMeta.content);
    }
  }

  // Only as a last resort - if neither document.title nor og:title had
  // anything usable - fall back to a real heading element, carefully
  // skipping nav/footer/header and hidden accessibility headings.
  if (!title) {
    title = firstVisibleHeadingOutsideChrome("h1") || firstVisibleHeadingOutsideChrome("h2");
  }

  // --- Metadata (experience level, hourly/fixed, duration, hours/week) ---
  // Upwork's own markup for these little badges changes often, so on top
  // of a few best-effort selectors, we also scan the page's visible text
  // for recognizable phrases Upwork consistently uses for each of these
  // fields, regardless of which element they're wrapped in.
  const metadataSelectors = [
    '[data-test="ContractorTier"]',
    '[data-test="JobTierText"]',
    '[data-test="job-type"]',
    '[data-test="Duration"]',
    '[data-test="features"]',
    '[class*="job-details" i]',
  ];
  const metadataFromSelectors = [];
  for (const selector of metadataSelectors) {
    document.querySelectorAll(selector).forEach((el) => {
      const text = el.innerText && el.innerText.trim();
      if (text && text.length < 80) metadataFromSelectors.push(text);
    });
  }

  const pageText = (document.querySelector("main") || document.body).innerText || "";
  const metadataPatterns = {
    "Experience level": /\b(Entry level|Intermediate|Expert)\b/i,
    "Pricing type": /\b(Hourly|Fixed-price|Fixed price)\b/i,
    Duration: /\b(Less than 1 month|1 to 3 months|3 to 6 months|More than 6 months)\b/i,
    "Hours per week": /(\d+\s*to\s*\d+\s*hrs\/week|\d+\+?\s*hrs\/week|Less than \d+\s*hrs\/week)/i,
  };
  const metadataFromText = [];
  for (const [label, pattern] of Object.entries(metadataPatterns)) {
    const match = pageText.match(pattern);
    if (match) metadataFromText.push(`${label}: ${match[0].trim()}`);
  }

  // Combine both sources and drop duplicates while keeping order.
  const seenMetadata = new Set();
  const metadata = [];
  for (const item of [...metadataFromSelectors, ...metadataFromText]) {
    if (!seenMetadata.has(item)) {
      seenMetadata.add(item);
      metadata.push(item);
    }
  }

  // Debug logging: open the browser DevTools console (F12) on a real
  // Upwork job page and click the extension button to see exactly what
  // was found - handy for spotting when a selector has gone stale.
  console.log("[Upwork Job Grabber] Title found:", title);
  console.log("[Upwork Job Grabber] Metadata found:", metadata);

  // --- Description ---
  // Try a handful of selectors that have historically matched Upwork's
  // job description block. These may drift out of date - that's fine,
  // it's just a best-effort list, not a requirement.
  const description = textFromFirstMatch([
    '[data-test="job-description-text"]',
    '[data-test="Description"]',
    '[data-test="job-description"]',
    ".job-description",
    '[class*="description"]',
  ]);

  // --- Skills / expertise section ---
  // Upwork typically lists required skills as a group of small "pill"
  // links/badges. We try a few likely containers and, if found, join
  // the individual skill labels back into a readable comma-separated
  // line instead of returning them jammed together.
  let skills = null;
  const skillsContainerSelectors = [
    '[data-test="Skills"]',
    '[data-test="skills-list"]',
    '[data-qa="skills"]',
    '[class*="skills"]',
  ];
  for (const selector of skillsContainerSelectors) {
    const container = document.querySelector(selector);
    if (container) {
      const items = Array.from(container.querySelectorAll("a, span, li"))
        .map((el) => el.innerText.trim())
        .filter((text) => text.length > 0 && text.length < 60); // skip long unrelated text
      if (items.length > 0) {
        skills = Array.from(new Set(items)).join(", "); // de-duplicate
        break;
      }
    }
  }

  // --- Assemble the final text ---
  // If we found at least the title or the description, build a clean,
  // labeled block from just those pieces (this is the "good" path).
  // Title goes first, then metadata, then description, then skills.
  if (title || description) {
    const parts = [];
    if (title) parts.push(title);
    if (metadata.length > 0) parts.push(metadata.join(" | "));
    if (description) parts.push(description);
    if (skills) parts.push(`Skills: ${skills}`);
    return parts.join("\n\n").trim();
  }

  // --- Fallback ---
  // None of our specific selectors matched anything (Upwork likely
  // changed their markup). Rather than fail, fall back to the page's
  // main visible text - better to hand match_llm.py something slightly
  // noisy than nothing at all.
  const main = document.querySelector("main");
  const fallbackText = (main ? main.innerText : document.body.innerText) || "";
  return fallbackText.trim();
}

// The value of this final expression becomes the result returned to
// chrome.scripting.executeScript()'s caller (see popup.js).
extractUpworkJobText();
