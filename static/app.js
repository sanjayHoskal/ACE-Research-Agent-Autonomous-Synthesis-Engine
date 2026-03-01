/* ═══════════════════════════════════════════════════════════════
   ACE Research Agent — Frontend Application Logic
   Handles research polling, Drafting Toggle, Citation Sidebar,
   and Hard-Reset context flush.
   ═══════════════════════════════════════════════════════════════ */

// ─── State ───────────────────────────────────────────────────────────────────
let currentView = "raw";
let pollingInterval = null;
let logOffset = 0;
let cachedResults = null;

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    mermaid.initialize({
        startOnLoad: false,
        theme: "dark",
        themeVariables: {
            primaryColor: "#8b5cf6",
            primaryTextColor: "#e8e8ed",
            primaryBorderColor: "#5b21b6",
            lineColor: "#8b5cf6",
            secondaryColor: "#22d3ee",
            tertiaryColor: "#16161f",
        },
    });

    // Enter key to launch
    document.getElementById("queryInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter") startResearch();
    });
});


// ═══ 1. START RESEARCH ═══════════════════════════════════════════════════════

async function startResearch() {
    const queryEl = document.getElementById("queryInput");
    const depthEl = document.getElementById("depthSelect");
    const query = queryEl.value.trim();

    if (!query) {
        queryEl.focus();
        queryEl.style.borderColor = "#f87171";
        setTimeout(() => queryEl.style.borderColor = "", 1500);
        return;
    }

    // Disable UI
    document.getElementById("launchBtn").disabled = true;
    document.getElementById("resetBtn").disabled = true;

    // Show progress & content
    document.getElementById("progressContainer").style.display = "block";
    document.getElementById("contentLayout").style.display = "grid";

    // Clear previous data
    clearResults();
    logOffset = 0;
    updateStatus("running", "Initializing...");
    updateProgress(2, "Launching mission...");

    try {
        const resp = await fetch("/api/research", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                search_depth: depthEl.value,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || "Failed to start research");
        }

        // Start polling
        startPolling();
    } catch (err) {
        updateStatus("error", err.message);
        document.getElementById("launchBtn").disabled = false;
        document.getElementById("resetBtn").disabled = false;
    }
}


// ═══ 2. POLLING ══════════════════════════════════════════════════════════════

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(pollStatus, 1500);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function pollStatus() {
    try {
        const resp = await fetch(`/api/status?since=${logOffset}`);
        const data = await resp.json();

        // Update logs
        if (data.logs && data.logs.length > 0) {
            const terminal = document.getElementById("logTerminal");
            for (const log of data.logs) {
                const entry = document.createElement("div");
                entry.className = "log-entry";
                entry.innerHTML = `<span class="log-time">${log.timestamp}</span>${escapeHtml(log.message)}`;
                terminal.appendChild(entry);
            }
            terminal.scrollTop = terminal.scrollHeight;
            logOffset = data.log_offset;
            document.getElementById("logCount").textContent = logOffset;
        }

        // Progress simulation based on status
        if (data.status === "running") {
            const elapsed = data.started_at
                ? (Date.now() - new Date(data.started_at).getTime()) / 1000
                : 0;
            const simProgress = Math.min(90, 5 + elapsed * 0.5);
            updateProgress(simProgress, `Processing... (${Math.round(elapsed)}s)`);
        }

        // Mission complete
        if (data.status === "complete") {
            stopPolling();
            updateProgress(100, "Complete!");
            updateStatus("complete", "Mission Complete");
            document.getElementById("launchBtn").disabled = false;
            document.getElementById("resetBtn").disabled = false;
            loadResults();
        }

        // Error
        if (data.status === "error") {
            stopPolling();
            updateStatus("error", data.error || "Unknown error");
            document.getElementById("launchBtn").disabled = false;
            document.getElementById("resetBtn").disabled = false;
        }
    } catch (err) {
        console.error("Polling error:", err);
    }
}


// ═══ 3. LOAD & RENDER RESULTS ════════════════════════════════════════════════

async function loadResults() {
    try {
        const resp = await fetch("/api/results");
        if (!resp.ok) return;
        cachedResults = await resp.json();
        renderResults(cachedResults);
    } catch (err) {
        console.error("Failed to load results:", err);
    }
}

function renderResults(data) {
    // Meta
    document.getElementById("resultsMeta").textContent =
        `${data.entity_count} entities · ${data.relationship_count} relationships · ${data.contradiction_count} contradictions`;

    // Stats row
    document.getElementById("statsRow").innerHTML = `
        <div class="stat-card">
            <div class="stat-value">${data.entity_count}</div>
            <div class="stat-label">Entities</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.relationship_count}</div>
            <div class="stat-label">Relationships</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.contradiction_count}</div>
            <div class="stat-label">Contradictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${data.sources.length}</div>
            <div class="stat-label">Sources</div>
        </div>
    `;

    // Entities
    renderEntities(data.entities);

    // Relationships
    renderRelationships(data.relationships);

    // Contradictions
    renderContradictions(data.contradictions);

    // Mermaid diagram
    renderMermaid(data.entities, data.relationships);

    // Whitepaper
    renderWhitepaper(data.report_markdown, data.sources);
}

function renderEntities(entities) {
    const grid = document.getElementById("entityGrid");
    // Sort: Protocol first, then by confidence
    const sorted = [...entities].sort((a, b) => {
        if (a.type === "Protocol" && b.type !== "Protocol") return -1;
        if (b.type === "Protocol" && a.type !== "Protocol") return 1;
        return b.confidence - a.confidence;
    });

    grid.innerHTML = sorted.map(e => `
        <div class="entity-card" data-entity-id="${e.id}" id="entity-${e.id}">
            <div>
                <span class="entity-name">${escapeHtml(e.name)}</span>
                <span class="entity-type">${escapeHtml(e.type)}</span>
            </div>
            <div class="entity-desc">${escapeHtml(e.description)}</div>
            <div class="entity-confidence">Confidence: ${Math.round(e.confidence * 100)}%</div>
        </div>
    `).join("");
}

function renderRelationships(relationships) {
    const wrap = document.getElementById("relationshipTable");
    if (!relationships.length) {
        wrap.innerHTML = '<p class="empty-state">No relationships detected.</p>';
        return;
    }

    // Filter out word-salad targets
    const wordSalad = new Set(["byte", "yes", "no", "in", "the", "max", "full", "low", "default", "on", "off", "su", "aifs"]);
    const filtered = relationships.filter(r =>
        !wordSalad.has(r.source.toLowerCase()) && !wordSalad.has(r.target.toLowerCase())
    );

    const getBadgeClass = (type) => {
        if (type.includes("competes")) return "competes";
        if (type.includes("requires")) return "requires";
        if (type.includes("implements")) return "implements";
        return "default";
    };

    wrap.innerHTML = `
        <table class="rel-table">
            <thead>
                <tr><th>Source</th><th>Relationship</th><th>Target</th></tr>
            </thead>
            <tbody>
                ${filtered.slice(0, 30).map(r => `
                    <tr>
                        <td>${escapeHtml(r.source)}</td>
                        <td><span class="rel-type-badge ${getBadgeClass(r.type)}">${escapeHtml(r.type.replace(/_/g, " "))}</span></td>
                        <td>${escapeHtml(r.target)}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

function renderContradictions(contradictions) {
    const list = document.getElementById("contradictionsList");
    if (!contradictions.length) {
        list.innerHTML = '<p class="empty-state">No contradictions detected.</p>';
        return;
    }

    list.innerHTML = contradictions.map(c => `
        <div class="contradiction-card ${c.resolved ? 'resolved' : ''}">
            <div class="contradiction-entity">
                ${escapeHtml(c.entity)}
                ${c.resolved ? '<span class="resolved-badge">✅ Resolved</span>' : ''}
            </div>
            <div class="claim-row">
                <div class="claim-label">Claim A:</div>
                <div class="claim-text">${escapeHtml(c.claim_a.substring(0, 200))}</div>
            </div>
            <div class="claim-row">
                <div class="claim-label">Claim B:</div>
                <div class="claim-text">${escapeHtml(c.claim_b.substring(0, 200))}</div>
            </div>
            ${c.resolution ? `<div class="resolution-text">${escapeHtml(c.resolution)}</div>` : ''}
        </div>
    `).join("");
}

async function renderMermaid(entities, relationships) {
    const container = document.getElementById("mermaidContainer");

    // Build Mermaid code from entities and relationships 
    const wordSalad = new Set(["byte", "yes", "no", "in", "the", "max", "full", "low", "default", "on", "off", "su", "aifs", "gbps", "ghz", "fi", "mcs", "nss", "sta", "cts", "mpdu", "mld", "traffic", "packet", "throughput"]);

    const protocolEntities = entities.filter(e =>
        e.type === "Protocol" && !wordSalad.has(e.name.toLowerCase())
    );

    const cleanRels = relationships.filter(r =>
        !wordSalad.has(r.source.toLowerCase()) && !wordSalad.has(r.target.toLowerCase()) &&
        r.source.length > 2 && r.target.length > 2
    );

    if (protocolEntities.length === 0) {
        container.innerHTML = '<p class="empty-state">No graph to display.</p>';
        return;
    }

    let mermaidCode = "graph TD\n";
    const nodeIds = new Map();
    let idCounter = 0;

    const getNodeId = (name) => {
        const key = name.toLowerCase();
        if (!nodeIds.has(key)) {
            nodeIds.set(key, `n${idCounter++}`);
        }
        return nodeIds.get(key);
    };

    // Add protocol nodes
    for (const e of protocolEntities) {
        const nid = getNodeId(e.name);
        mermaidCode += `    ${nid}["${e.name}"]\n`;
    }

    // Add edges
    const usedEdges = new Set();
    for (const r of cleanRels.slice(0, 20)) {
        const srcId = getNodeId(r.source);
        const tgtId = getNodeId(r.target);
        const edgeKey = `${srcId}-${tgtId}`;
        if (usedEdges.has(edgeKey)) continue;
        usedEdges.add(edgeKey);
        const label = r.type.replace(/_/g, " ");
        mermaidCode += `    ${srcId} -->|${label}| ${tgtId}\n`;
    }

    // Style protocol nodes
    for (const e of protocolEntities) {
        const nid = getNodeId(e.name);
        mermaidCode += `    style ${nid} fill:#1e1e2e,stroke:#8b5cf6,stroke-width:2px,color:#e8e8ed\n`;
    }

    try {
        container.innerHTML = "";
        const { svg } = await mermaid.render("kg-diagram", mermaidCode);
        container.innerHTML = svg;
    } catch (err) {
        container.innerHTML = `<pre style="color:var(--text-muted);font-size:0.75rem;">${escapeHtml(mermaidCode)}</pre>`;
    }
}

function renderWhitepaper(reportMd, sources) {
    const container = document.getElementById("whitepaperRendered");

    if (!reportMd) {
        container.innerHTML = '<p class="empty-state">No whitepaper generated yet.</p>';
        return;
    }

    // Render markdown
    let html = marked.parse(reportMd);

    // Make [Source N] clickable citations
    html = html.replace(
        /\[Source\s+(\d+)\]/g,
        '<span class="citation-link" onclick="openCitation($1)">[Source $1]</span>'
    );

    // Also make source links in the report clickable for the sidebar
    if (sources && sources.length > 0) {
        for (const src of sources) {
            // Match numbered references like [1], [2] etc that link to source URLs
            const urlShort = src.url.substring(0, 40);
            html = html.replace(
                new RegExp(`\\[${src.id}\\]\\(${escapeRegex(src.url)}\\)`, 'g'),
                `<span class="citation-link" onclick="openCitation(${src.id})">[${src.id}]</span>`
            );
        }
    }

    container.innerHTML = html;

    // Re-render mermaid blocks inside the whitepaper
    container.querySelectorAll("pre code.language-mermaid").forEach(async (el, idx) => {
        const code = el.textContent;
        try {
            const { svg } = await mermaid.render(`wp-mermaid-${idx}`, code);
            el.parentElement.outerHTML = `<div class="mermaid-container">${svg}</div>`;
        } catch (e) { /* keep as code block */ }
    });
}


// ═══ 4. DRAFTING TOGGLE ══════════════════════════════════════════════════════

function toggleView(view) {
    currentView = view;

    // Update tab styles
    document.querySelectorAll(".tab").forEach(t => {
        t.classList.toggle("active", t.dataset.tab === view);
    });

    // Show/hide content
    document.getElementById("rawView").classList.toggle("active", view === "raw");
    document.getElementById("whitepaperView").classList.toggle("active", view === "whitepaper");
}


// ═══ 5. CITATION SIDEBAR ═════════════════════════════════════════════════════

async function openCitation(sourceId) {
    const sidebar = document.getElementById("citationSidebar");
    const overlay = document.getElementById("citationOverlay");
    const content = document.getElementById("sidebarContent");

    // Show sidebar
    sidebar.classList.add("active");
    overlay.classList.add("active");

    content.innerHTML = '<p class="empty-state">Loading...</p>';

    try {
        const resp = await fetch(`/api/citation/${sourceId}`);
        const data = await resp.json();

        if (data.error) {
            content.innerHTML = `<p class="empty-state">${escapeHtml(data.error)}</p>`;
            return;
        }

        let citingHtml = data.citing_entities.map(e =>
            `<span class="sidebar-entity-chip" onclick="highlightEntity('${e.id}')">${escapeHtml(e.name)} <span style="opacity:0.5">(${e.type})</span></span>`
        ).join("");

        if (!citingHtml) {
            citingHtml = '<span style="color:var(--text-muted);font-size:0.8rem;">No KG nodes directly cite this source.</span>';
        }

        content.innerHTML = `
            <div class="sidebar-section-title">Source URL</div>
            <div class="sidebar-url">
                <a href="${escapeHtml(data.url)}" target="_blank" rel="noopener">${escapeHtml(data.url)}</a>
            </div>
            <div class="sidebar-section-title">Knowledge Graph Nodes (${data.citing_entities.length})</div>
            <div style="margin-top:4px;">
                ${citingHtml}
            </div>
        `;

        // Highlight all citing entities in the KG
        clearHighlights();
        for (const e of data.citing_entities) {
            const card = document.getElementById(`entity-${e.id}`);
            if (card) {
                card.classList.add("highlighted");
                // Scroll into view if on Raw tab
                if (currentView === "raw") {
                    card.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            }
        }
    } catch (err) {
        content.innerHTML = `<p class="empty-state">Error loading citation details.</p>`;
    }
}

function highlightEntity(entityId) {
    // Switch to raw view and scroll to entity
    toggleView("raw");
    clearHighlights();

    const card = document.getElementById(`entity-${entityId}`);
    if (card) {
        card.classList.add("highlighted");
        card.scrollIntoView({ behavior: "smooth", block: "center" });
    }
}

function clearHighlights() {
    document.querySelectorAll(".entity-card.highlighted").forEach(el => {
        el.classList.remove("highlighted");
    });
}

function closeSidebar() {
    document.getElementById("citationSidebar").classList.remove("active");
    document.getElementById("citationOverlay").classList.remove("active");
    clearHighlights();
}


// ═══ 6. HARD-RESET ═══════════════════════════════════════════════════════════

async function hardReset() {
    stopPolling();

    try {
        await fetch("/api/reset", { method: "POST" });
    } catch (err) {
        // OK if server not running
    }

    // Clear frontend state
    logOffset = 0;
    cachedResults = null;
    currentView = "raw";

    // Clear UI
    clearResults();
    document.getElementById("queryInput").value = "";
    document.getElementById("logTerminal").innerHTML =
        '<div class="log-entry log-system">Context flushed — ready for new mission.</div>';
    document.getElementById("logCount").textContent = "0";
    document.getElementById("progressContainer").style.display = "none";
    document.getElementById("contentLayout").style.display = "none";

    updateStatus("idle", "Idle");

    document.getElementById("launchBtn").disabled = false;
    document.getElementById("resetBtn").disabled = false;

    // Focus query input
    document.getElementById("queryInput").focus();
}


// ═══ UTILITIES ═══════════════════════════════════════════════════════════════

function clearResults() {
    document.getElementById("statsRow").innerHTML = "";
    document.getElementById("entityGrid").innerHTML = "";
    document.getElementById("relationshipTable").innerHTML = "";
    document.getElementById("contradictionsList").innerHTML = "";
    document.getElementById("mermaidContainer").innerHTML =
        '<p class="empty-state">Waiting for data...</p>';
    document.getElementById("whitepaperRendered").innerHTML =
        '<p class="empty-state">Run a research mission to generate the whitepaper.</p>';
    document.getElementById("resultsMeta").textContent = "";

    // Reset tabs
    toggleView("raw");
}

function updateStatus(status, text) {
    const badge = document.getElementById("statusBadge");
    badge.className = `status-badge ${status}`;
    badge.querySelector(".status-text").textContent = text;
}

function updateProgress(percent, label) {
    document.getElementById("progressFill").style.width = `${percent}%`;
    document.getElementById("progressLabel").textContent = label;
}

function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
