// Minimal chat UI for the Clinical Knowledge Graph QA API.
// Talks to the same origin that serves this page: GET /health, POST /ask.

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addFacts(container, facts) {
  if (!facts || !facts.length) return;
  const d = document.createElement("details");
  d.className = "facts";
  const s = document.createElement("summary");
  s.textContent = `${facts.length} graph fact(s) used`;
  d.appendChild(s);
  const ul = document.createElement("ul");
  for (const f of facts) {
    const li = document.createElement("li");
    li.textContent = f;
    ul.appendChild(li);
  }
  d.appendChild(ul);
  container.appendChild(d);
}

async function ask(question) {
  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  const bot = addMessage("…", "bot");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    bot.textContent = data.answer || "(no answer)";
    addFacts(bot, data.facts);
  } catch (err) {
    bot.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) ask(q);
});

document.getElementById("examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

// Populate the health pills on load.
fetch("/health")
  .then((r) => r.json())
  .then((h) => {
    document.getElementById("backend").textContent = `backend: ${h.backend}`;
    document.getElementById("nodes").textContent = `graph: ${h.graph_nodes} nodes`;
  })
  .catch(() => {});
