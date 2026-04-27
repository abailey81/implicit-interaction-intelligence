/* Iter 51 (2026-04-27) — wires the 7 new Huawei dashboard tabs.
 *
 * Tabs:
 *   #intent       — POST /api/intent + GET /api/intent/status
 *   #edge-profile — GET /api/profiling/report
 *   #finetune     — GET /api/intent/status (renders side-by-side)
 *   #facts        — GET /api/v1/facts (via WS state_update); forget button POSTs
 *   #multimodal   — static (markup in index.html)
 *   #research     — fetches docs/huawei/research_reading_list.md, renders
 *   #jdmap        — fetches docs/huawei/jd_to_repo_map.md, renders
 *
 * Loaded after dashboard.js so the existing tab router catches the new
 * data-tab values automatically (no router code changes needed).
 */
(function () {
    'use strict';

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    function getJSON(url) {
        return fetch(url, {credentials: 'same-origin'})
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            });
    }

    function postJSON(url, body) {
        return fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            credentials: 'same-origin',
            body: JSON.stringify(body),
        }).then(function (r) {
            return r.json().then(function (data) {
                return {status: r.status, data: data};
            });
        });
    }

    function escHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function renderJSON(obj) {
        var json = JSON.stringify(obj, null, 2);
        // Lightweight syntax highlight.
        return '<pre class="intent-json-block">' +
            escHtml(json)
                .replace(/&quot;([^&]+?)&quot;:/g, '<span class="json-key">"$1"</span>:')
                .replace(/: &quot;([^&]*?)&quot;/g, ': <span class="json-string">"$1"</span>')
                .replace(/: (\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
                .replace(/: (true|false|null)/g, ': <span class="json-bool">$1</span>')
            + '</pre>';
    }

    // -------------------------------------------------------------------
    // Intent tab
    // -------------------------------------------------------------------

    function wireIntentTab() {
        var input = document.getElementById('intent-input');
        var backendSel = document.getElementById('intent-backend');
        var btn = document.getElementById('intent-submit');
        var out = document.getElementById('intent-output');
        var meta = document.getElementById('intent-meta');
        if (!input || !btn || !out) return;

        function submit() {
            var text = input.value.trim();
            if (!text) return;
            var backend = backendSel.value;
            out.classList.remove('error');
            out.innerHTML = '<em>Parsing via ' + backend + '…</em>';
            meta.innerHTML = '';
            postJSON('/api/intent', {text: text, backend: backend})
                .then(function (resp) {
                    var d = resp.data;
                    if (d.error && !d.action) {
                        out.classList.add('error');
                        out.textContent = d.error;
                    } else {
                        out.innerHTML = renderJSON({
                            action: d.action,
                            params: d.params,
                            valid_json: d.valid_json,
                            valid_action: d.valid_action,
                            valid_slots: d.valid_slots,
                            confidence: d.confidence,
                        });
                    }
                    var chips = [];
                    chips.push('<span class="' + (d.valid_json ? 'meta-good' : 'meta-bad') +
                               '">json ' + (d.valid_json ? 'valid' : 'invalid') + '</span>');
                    chips.push('<span class="' + (d.valid_action ? 'meta-good' : 'meta-bad') +
                               '">action ' + (d.valid_action ? 'ok' : 'unknown') + '</span>');
                    chips.push('<span class="' + (d.valid_slots ? 'meta-good' : 'meta-warn') +
                               '">slots ' + (d.valid_slots ? 'ok' : 'check') + '</span>');
                    chips.push('<span>backend ' + escHtml(d.backend) + '</span>');
                    chips.push('<span>' + (d.latency_ms || 0).toFixed(1) + ' ms</span>');
                    if (d.error) chips.push('<span class="meta-warn">err: ' + escHtml(d.error) + '</span>');
                    meta.innerHTML = chips.join('');
                })
                .catch(function (err) {
                    out.classList.add('error');
                    out.textContent = 'Request failed: ' + err.message;
                });
        }

        btn.addEventListener('click', submit);
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') submit();
        });

        // Status refresh (lazy on first reveal of the tab).
        var statusEl = document.getElementById('intent-status-block');
        function refreshStatus() {
            getJSON('/api/intent/status')
                .then(function (data) {
                    statusEl.innerHTML = '<code>' + escHtml(JSON.stringify(data, null, 2)) + '</code>';
                })
                .catch(function (err) {
                    statusEl.innerHTML = '<code>error: ' + escHtml(err.message) + '</code>';
                });
        }
        refreshStatus();
        // Refresh when the tab is shown.
        document.addEventListener('tab:shown', function (e) {
            if (e && e.detail && e.detail.tab === 'intent') refreshStatus();
        });
    }

    // -------------------------------------------------------------------
    // Edge Profile tab
    // -------------------------------------------------------------------

    function renderEdgeProfile(data) {
        var container = document.getElementById('edge-profile-render');
        if (!container) return;
        var rows = (data.components || []).map(function (c) {
            return (
                '<tr>' +
                '<td>' + escHtml(c.name) + '</td>' +
                '<td class="num">' + (c.params_m || 0).toFixed(2) + '</td>' +
                '<td class="num">' + (c.fp32_mb || 0).toFixed(2) + '</td>' +
                '<td class="num">' + (c.int8_mb || 0).toFixed(2) + '</td>' +
                '<td class="num">' + (c.p50_ms || 0).toFixed(1) + '</td>' +
                '</tr>'
            );
        }).join('');
        var fits = data.fits_budget;
        container.innerHTML =
            '<div class="edge-profile-summary">' +
              '<div class="stat"><div class="stat-label">Total latency P50</div>' +
              '<div class="stat-value ' + (fits ? 'good' : '') + '">' +
                (data.total_latency_ms || 0).toFixed(1) + ' ms</div></div>' +
              '<div class="stat"><div class="stat-label">Memory (INT8)</div>' +
              '<div class="stat-value">' + (data.memory_mb || 0).toFixed(1) + ' MB</div></div>' +
              '<div class="stat"><div class="stat-label">Budget</div>' +
              '<div class="stat-value">' + (data.budget_ms || 0).toFixed(0) + ' ms</div></div>' +
              '<div class="stat"><div class="stat-label">Fits budget?</div>' +
              '<div class="stat-value ' + (fits ? 'good' : '') + '">' +
                (fits ? '✓ YES' : '✗ no') + '</div></div>' +
              '<div class="stat"><div class="stat-label">Device class</div>' +
              '<div class="stat-value" style="font-size:13px;">' +
                escHtml(data.device_class || 'n/a') + '</div></div>' +
            '</div>' +
            '<table class="edge-profile-table">' +
            '<thead><tr><th>Component</th><th class="num">Params (M)</th>' +
            '<th class="num">FP32 MB</th><th class="num">INT8 MB</th>' +
            '<th class="num">P50 ms</th></tr></thead>' +
            '<tbody>' + rows + '</tbody></table>';
    }

    function wireEdgeProfileTab() {
        function refresh() {
            getJSON('/api/profiling/report')
                .then(renderEdgeProfile)
                .catch(function (err) {
                    var c = document.getElementById('edge-profile-render');
                    if (c) c.innerHTML = '<em>Failed to load: ' + escHtml(err.message) + '</em>';
                });
        }
        refresh();
        document.addEventListener('tab:shown', function (e) {
            if (e && e.detail && e.detail.tab === 'edge-profile') refresh();
        });
    }

    // -------------------------------------------------------------------
    // Fine-tune Comparison tab
    // -------------------------------------------------------------------

    function renderFinetuneComparison(data) {
        var container = document.getElementById('finetune-comparison');
        if (!container) return;
        function rowFor(label, qwen, gemini) {
            return '<tr><td>' + escHtml(label) + '</td><td>' +
                escHtml(qwen) + '</td><td>' + escHtml(gemini) + '</td></tr>';
        }
        var qwen = data.qwen || {};
        var gem = data.gemini || {};
        var qmet = qwen.training_metrics || {};
        var qeval = (data.eval || {}).qwen || {};
        var gemeval = (data.eval || {}).gemini || {};
        var html =
            '<table class="edge-profile-table">' +
            '<thead><tr><th></th><th>Open-weight (Qwen LoRA)</th><th>Cloud (Gemini AI Studio)</th></tr></thead>' +
            '<tbody>' +
              rowFor('Adapter loaded?', qwen.ready ? '✓ yes' : '✗ not loaded',
                                       gem.ready ? '✓ yes' : '✗ not configured') +
              rowFor('Base model', qmet.model || '?', '(Gemini 2.5 Flash)') +
              rowFor('Trainable params (M)',
                     qmet.use_dora ? 'DoRA r=' + (qmet.rank || '?') : 'LoRA r=' + (qmet.rank || '?'),
                     'Vendor-managed') +
              rowFor('Best val_loss', qmet.best_val_loss != null ? qmet.best_val_loss.toFixed(4) : '?',
                                     '?') +
              rowFor('Wall time (s)', qmet.wall_time_s != null ? qmet.wall_time_s.toFixed(0) : '?',
                                      gem.tuning_result ? (gem.tuning_result.wall_time_s || '?') + '' : '?') +
              rowFor('Action accuracy (test)',
                     qeval.action_accuracy != null ? (qeval.action_accuracy * 100).toFixed(1) + '%' : '—',
                     gemeval.action_accuracy != null ? (gemeval.action_accuracy * 100).toFixed(1) + '%' : '—') +
              rowFor('Full-match (test)',
                     qeval.full_match_rate != null ? (qeval.full_match_rate * 100).toFixed(1) + '%' : '—',
                     gemeval.full_match_rate != null ? (gemeval.full_match_rate * 100).toFixed(1) + '%' : '—') +
              rowFor('Latency P50 (ms)',
                     qeval.latency_p50_ms != null ? qeval.latency_p50_ms.toFixed(1) : '—',
                     gemeval.latency_p50_ms != null ? gemeval.latency_p50_ms.toFixed(1) : '—') +
              rowFor('Cost', '£0 (laptop electricity)',
                     'Free tier — ~£0 per fine-tune; ~£0.00001 per call') +
              rowFor('Privacy', 'On-device only', 'Utterance leaves device per call') +
              rowFor('Deployable on Kirin?', '✓ via MindSpore Lite', '✗ vendor-hosted only') +
            '</tbody></table>';
        container.innerHTML = html;
    }

    function wireFinetuneTab() {
        function refresh() {
            getJSON('/api/intent/status')
                .then(renderFinetuneComparison)
                .catch(function (err) {
                    var c = document.getElementById('finetune-comparison');
                    if (c) c.innerHTML = '<em>Failed to load: ' + escHtml(err.message) + '</em>';
                });
        }
        refresh();
        document.addEventListener('tab:shown', function (e) {
            if (e && e.detail && e.detail.tab === 'finetune') refresh();
        });
    }

    // -------------------------------------------------------------------
    // Personal Facts tab
    // -------------------------------------------------------------------

    function renderFacts(facts) {
        var el = document.getElementById('facts-list');
        if (!el) return;
        if (!facts || Object.keys(facts).length === 0) {
            el.innerHTML = '<em>No facts stored yet.  Tell the model "my name is X", "I live in Y" via the Chat tab — values appear here on the next turn.</em>';
            return;
        }
        var rows = Object.keys(facts).sort().map(function (k) {
            return '<tr><td><strong>' + escHtml(k) + '</strong></td><td>' +
                escHtml(facts[k]) + '</td></tr>';
        }).join('');
        el.innerHTML = '<table class="edge-profile-table"><thead><tr><th>Slot</th><th>Value</th></tr></thead>' +
            '<tbody>' + rows + '</tbody></table>';
    }

    function wireFactsTab() {
        // The pipeline doesn't yet expose a /facts endpoint; we listen
        // to WS state_update frames where the engine ships personal_facts
        // metadata.  Hook into the existing WebSocket bus.
        document.addEventListener('i3:state_update', function (e) {
            if (e && e.detail && e.detail.personal_facts) {
                renderFacts(e.detail.personal_facts);
            }
        });
        // Initial empty render.
        renderFacts({});

        var btn = document.getElementById('facts-forget-button');
        if (btn) {
            btn.addEventListener('click', function () {
                if (!confirm('Forget every personal fact in this session AND wipe the encrypted DB rows?')) return;
                // Send a chat message via the existing WS chat.
                if (window.i3 && window.i3.sendChatMessage) {
                    window.i3.sendChatMessage('forget my facts');
                } else {
                    alert('Type "forget my facts" in the chat to wipe.');
                }
            });
        }
    }

    // -------------------------------------------------------------------
    // Research Notes tab — render docs/huawei/research_reading_list.md
    // -------------------------------------------------------------------

    function wireResearchTab() {
        function load() {
            fetch('/static/docs/huawei/research_reading_list.md', {credentials: 'same-origin'})
                .then(function (r) { return r.text(); })
                .then(function (md) {
                    var grid = document.getElementById('research-grid');
                    if (!grid) return;
                    // Lightweight markdown → cards: split on "### N. " headers.
                    var sections = md.split(/^### /m).slice(1);
                    var cards = sections.slice(0, 20).map(function (sec) {
                        var nl = sec.indexOf('\n');
                        var title = sec.slice(0, nl).trim();
                        var rest = sec.slice(nl + 1).trim();
                        // Truncate rest to ~250 chars.
                        var snip = rest.length > 250 ? rest.slice(0, 250) + '…' : rest;
                        return '<div class="huawei-card"><h3>' + escHtml(title) + '</h3>' +
                            '<p>' + escHtml(snip).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>') + '</p></div>';
                    });
                    grid.innerHTML = cards.join('');
                })
                .catch(function () {
                    var grid = document.getElementById('research-grid');
                    if (grid) grid.innerHTML = '<em>Could not load research_reading_list.md</em>';
                });
        }
        document.addEventListener('tab:shown', function (e) {
            if (e && e.detail && e.detail.tab === 'research') load();
        });
    }

    // -------------------------------------------------------------------
    // JD Map tab — render docs/huawei/jd_to_repo_map.md
    // -------------------------------------------------------------------

    function wireJdmapTab() {
        function load() {
            fetch('/static/docs/huawei/jd_to_repo_map.md', {credentials: 'same-origin'})
                .then(function (r) { return r.text(); })
                .then(function (md) {
                    var c = document.getElementById('jdmap-render');
                    if (!c) return;
                    // Very light markdown render: headings, links, tables.
                    var html = escHtml(md)
                        .replace(/^## (.+)$/gm, '<h3 style="margin-top:18px;color:#fff;">$1</h3>')
                        .replace(/^### (.+)$/gm, '<h4 style="margin-top:14px;color:#ddd;">$1</h4>')
                        .replace(/\[([^\]]+)\]\(([^)]+)\)/g,
                                 '<a href="$2" target="_blank">$1</a>')
                        .replace(/`([^`]+)`/g,
                                 '<code style="background:#1a1a1c;padding:2px 5px;border-radius:3px;font-size:12px;">$1</code>')
                        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                        .replace(/\n/g, '<br>');
                    c.innerHTML = '<div style="font-size:13px;line-height:1.5;">' + html + '</div>';
                })
                .catch(function () {
                    var c = document.getElementById('jdmap-render');
                    if (c) c.innerHTML = '<em>Could not load jd_to_repo_map.md</em>';
                });
        }
        document.addEventListener('tab:shown', function (e) {
            if (e && e.detail && e.detail.tab === 'jdmap') load();
        });
    }

    // -------------------------------------------------------------------
    // Wire everything on DOM ready
    // -------------------------------------------------------------------

    function init() {
        wireIntentTab();
        wireEdgeProfileTab();
        wireFinetuneTab();
        wireFactsTab();
        wireResearchTab();
        wireJdmapTab();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
