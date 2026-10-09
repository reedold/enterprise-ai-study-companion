"use strict";

// Same origin when served by FastAPI; fall back to local dev server when opened as a file.
const API = location.protocol === "file:" ? "http://127.0.0.1:8000/api/v1" : "/api/v1";
const MAX_MB = 25;
const SUBJECTS_KEY = "ke.subjects";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ helpers ---
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function showMsg(node, text, kind) {
  node.textContent = text;
  node.className = `msg msg--${kind}`;
  node.hidden = false;
}

function setBusy(btn, busy, label) {
  btn.disabled = busy;
  btn.classList.toggle("is-busy", busy);
  if (label) btn.querySelector(".btn__label").textContent = label;
}

function errorText(status, body) {
  const message = body?.error?.message || body?.detail || `Request failed (HTTP ${status})`;
  const details = body?.error?.details;
  if (Array.isArray(details) && details.length) {
    const first = details[0];
    return `${message}: ${(first.loc || []).slice(1).join(".")} ${first.msg || ""}`.trim();
  }
  return message;
}

function parseJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

// --------------------------------------------------- remembered subjects ---
function loadSubjects() {
  try { return JSON.parse(localStorage.getItem(SUBJECTS_KEY)) || []; } catch { return []; }
}

function rememberSubject(subject) {
  if (!subject) return;
  const list = [subject, ...loadSubjects().filter((s) => s !== subject)].slice(0, 15);
  try { localStorage.setItem(SUBJECTS_KEY, JSON.stringify(list)); } catch { /* storage unavailable */ }
  renderSubjects();
}

function renderSubjects() {
  const subjects = loadSubjects();
  $("subject-list").replaceChildren(...subjects.map((s) => el("option", { value: s })));
  if (!$("up-subject").value && subjects[0]) $("up-subject").value = subjects[0];
  if (!$("gen-subject").value && subjects[0]) $("gen-subject").value = subjects[0];
}

// ---------------------------------------------------------- health pill ---
async function checkHealth() {
  const pill = $("status");
  try {
    const res = await fetch(`${API}/health`, { cache: "no-store" });
    if (!res.ok) throw new Error();
    pill.className = "status status--ok";
    pill.querySelector(".label").textContent = "API online";
  } catch {
    pill.className = "status status--down";
    pill.querySelector(".label").textContent = "API offline";
  }
}

// --------------------------------------------------------------- upload ---
let selectedFile = null;

function selectFile(file) {
  const zone = $("dropzone");
  const msg = $("upload-msg");
  msg.hidden = true;
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showMsg(msg, "Only PDF files are supported.", "error");
    return;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    showMsg(msg, `File is larger than ${MAX_MB} MB.`, "error");
    return;
  }
  selectedFile = file;
  zone.classList.add("has-file");
  $("dropzone-text").replaceChildren(
    el("strong", { text: file.name }),
    ` · ${(file.size / 1024 / 1024).toFixed(2)} MB`,
  );
  updateUploadButton();
}

function updateUploadButton() {
  $("upload-btn").disabled = !(selectedFile && $("up-subject").value.trim());
}

function setupDropzone() {
  const zone = $("dropzone");
  const input = $("file-input");
  input.addEventListener("change", () => selectFile(input.files[0]));
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("is-drag"); }));
  zone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
}

function uploadPdf(event) {
  event.preventDefault();
  const subject = $("up-subject").value.trim();
  const topic = $("up-topic").value.trim();
  const btn = $("upload-btn");
  const msg = $("upload-msg");
  const progress = $("upload-progress");
  const bar = progress.querySelector(".progress__bar");
  if (!selectedFile || !subject) return;

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("subject_id", subject);
  if (topic) form.append("topic_name", topic);

  // XHR (not fetch) so we can show real upload progress.
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${API}/upload-pdf`);
  setBusy(btn, true, "Uploading…");
  progress.hidden = false;
  progress.classList.remove("is-indeterminate");
  bar.style.width = "0%";
  msg.hidden = true;

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) bar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
  };
  xhr.upload.onload = () => {
    progress.classList.add("is-indeterminate");
    bar.style.width = "";
    setBusy(btn, true, "Processing…");
    showMsg(msg, "Extracting text, embedding and indexing. Scanned PDFs need OCR and can take a while.", "info");
  };
  xhr.onload = () => {
    const body = parseJson(xhr.responseText);
    if (xhr.status >= 200 && xhr.status < 300 && body) {
      const ocr = body.ocr_pages ? ` (${body.ocr_pages} via OCR)` : "";
      showMsg(msg, `Indexed “${body.filename}”: ${body.pages} pages${ocr} → ${body.chunks_stored} chunks.`, "ok");
      rememberSubject(subject);
      $("gen-subject").value = subject;
    } else {
      showMsg(msg, errorText(xhr.status, body), "error");
    }
    done();
  };
  xhr.onerror = () => { showMsg(msg, "Network error — is the API running?", "error"); done(); };

  function done() {
    progress.hidden = true;
    setBusy(btn, false, "Upload & index");
    updateUploadButton();
  }
  xhr.send(form);
}

// ------------------------------------------------------------- generate ---
const loadingSteps = ["Searching your documents…", "Reading the most relevant passages…", "Writing your study pack…"];
let loadingTimer = null;

function setView(view) {
  $("empty-state").hidden = view !== "empty";
  $("loading").hidden = view !== "loading";
  $("results").hidden = view !== "results";
  clearInterval(loadingTimer);
  if (view === "loading") {
    let i = 0;
    $("loading-text").textContent = loadingSteps[0];
    loadingTimer = setInterval(() => {
      i = Math.min(i + 1, loadingSteps.length - 1);
      $("loading-text").textContent = loadingSteps[i];
    }, 1800);
  }
}

async function generate(event) {
  event.preventDefault();
  const query = $("gen-query").value.trim();
  const msg = $("gen-msg");
  const btn = $("gen-btn");
  msg.hidden = true;
  if (query.length < 3) {
    showMsg(msg, "Enter a topic or question (at least 3 characters).", "error");
    return;
  }

  const payload = { query };
  const subject = $("gen-subject").value.trim();
  const topK = parseInt($("gen-topk").value, 10);
  if (subject) payload.subject_id = subject;
  if (topK >= 1 && topK <= 20) payload.top_k = topK;

  setBusy(btn, true, "Generating…");
  setView("loading");
  try {
    const res = await fetch(`${API}/generate-study-material`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = parseJson(await res.text());
    if (!res.ok || !body) throw new Error(errorText(res.status, body));
    renderResults(body);
    setView("results");
    if (subject) rememberSubject(subject);
    if (window.matchMedia("(max-width: 900px)").matches) {
      $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    setView("empty");
    showMsg(msg, err.message === "Failed to fetch" ? "Network error — is the API running?" : err.message, "error");
  } finally {
    setBusy(btn, false, "Generate");
  }
}

// --------------------------------------------------------- rendering ---
function citations(refs = []) {
  return refs.map((r) =>
    el("button", {
      class: "cite",
      type: "button",
      title: `Show source ${r}`,
      text: String(r),
      onclick: () => flashSource(r),
    }));
}

function flashSource(ref) {
  const node = document.querySelector(`.source[data-ref="${ref}"]`);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "center" });
  node.classList.add("is-flash");
  setTimeout(() => node.classList.remove("is-flash"), 1600);
}

function renderResults(data) {
  const grounded = data.grounded;
  $("not-grounded").hidden = grounded;
  $("grounded-content").hidden = !grounded;
  $("not-grounded-text").textContent = data.message || "";

  if (grounded) {
    $("res-title").textContent = data.query;
    $("res-summary").replaceChildren(data.summary, ...citations(data.summary_sources));

    $("concepts-section").hidden = !data.key_concepts.length;
    $("res-concepts").replaceChildren(...data.key_concepts.map((c) =>
      el("div", { class: "concept" },
        el("dt", {}, c.term, ...citations(c.sources)),
        el("dd", { text: c.explanation }))));

    $("formulas-section").hidden = !data.formulas.length;
    $("res-formulas").replaceChildren(...data.formulas.map((f) =>
      el("div", { class: "formula" },
        el("div", { class: "formula__name" }, f.name, ...citations(f.sources)),
        el("code", { class: "formula__expr", text: f.expression }),
        f.explanation ? el("div", { class: "formula__desc", text: f.explanation }) : null)));

    $("res-quiz").replaceChildren(...data.self_assessment.map((q) => {
      const li = el("li");
      li.append(
        el("button", {
          class: "quiz__q", type: "button", text: q.question,
          onclick: () => li.classList.toggle("is-open"),
        }),
        el("div", { class: "quiz__a" }, q.answer, ...citations(q.sources)));
      return li;
    }));
    $("reveal-all").textContent = "Reveal all";
  }

  $("sources-section").hidden = !data.sources.length;
  $("res-sources").replaceChildren(...data.sources.map((s) => {
    const page = s.page != null ? ` · page ${s.page}` : "";
    return el("li", { class: "source", "data-ref": String(s.ref) },
      el("span", { class: "source__ref", text: String(s.ref) }),
      el("div", { class: "source__meta" },
        `${s.filename || "Unknown file"}${page}`,
        el("span", { class: "muted", text: ` · ${Math.round(s.similarity * 100)}% match` })),
      el("div", { class: "source__excerpt", text: `“${s.excerpt}${s.excerpt.length >= 240 ? "…" : ""}”` }));
  }));
  $("res-model").textContent = `Generated by ${data.model}`;
}

function toggleRevealAll() {
  const items = [...$("res-quiz").children];
  const open = !items.every((li) => li.classList.contains("is-open"));
  items.forEach((li) => li.classList.toggle("is-open", open));
  $("reveal-all").textContent = open ? "Hide all" : "Reveal all";
}

// ----------------------------------------------------------------- init ---
setupDropzone();
renderSubjects();
$("up-subject").addEventListener("input", updateUploadButton);
$("upload-form").addEventListener("submit", uploadPdf);
$("gen-form").addEventListener("submit", generate);
$("gen-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("gen-form").requestSubmit();
});
$("reveal-all").addEventListener("click", toggleRevealAll);
checkHealth();
setInterval(checkHealth, 30000);
