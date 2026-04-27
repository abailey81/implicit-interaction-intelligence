/**
 * Hash-based tab router for the Apple-style multi-tab layout.
 *
 * Each navigable section has:
 *   - a nav link  ``<a class="nav-link" data-tab="X" href="#X">``
 *   - a panel     ``<section class="tab-panel" data-tab="X" hidden>``
 *
 * Two new structural pieces (post-2026-04-25 apple21 cleanup):
 *
 *   1. Nav dropdown groups — `<div class="nav-group" data-group="lab">`
 *      with a `.nav-group-trigger` button and a `.nav-group-menu`
 *      containing the actual `.nav-link` items.  When the user activates
 *      a tab whose link lives inside a group, we mark that group's
 *      trigger active so the parent label still highlights.  Hover/focus
 *      open the menu; clicking an item routes through the same hash
 *      pipeline as a top-level link.
 *
 * The router:
 *   1. Reads ``location.hash`` on load and on hashchange.
 *   2. Activates the matching tab + panel; hides the rest.
 *   3. Triggers a "reveal" animation pass on the active panel.
 *   4. Closes any open dropdown menu on outside click.
 *
 * No build step, no framework.  Plain DOM.
 */

(() => {
    "use strict";

    const DEFAULT_TAB = "chat";
    const VALID_TABS = new Set([
        "chat", "stack", "state", "adaptation",
        "routing", "flow", "playground", "privacy", "profile", "edge",
        "benchmarks", "design", "references", "about",
    ]);

    // Map tab → group name (for parent-trigger activation)
    const TAB_TO_GROUP = {
        flow: "lab",
        benchmarks: "lab",
        playground: "lab",
        design: "docs",
        references: "docs",
    };

    function getHashTab() {
        const raw = (location.hash || "").replace(/^#/, "").trim().toLowerCase();
        return VALID_TABS.has(raw) ? raw : DEFAULT_TAB;
    }

    function setActiveTab(name, { focusInput = false } = {}) {
        if (!VALID_TABS.has(name)) name = DEFAULT_TAB;

        // Toggle nav links — every link with a data-tab matches here.
        document.querySelectorAll(".nav-link").forEach((link) => {
            const isActive = link.dataset.tab === name;
            link.classList.toggle("active", isActive);
            if (link.getAttribute("role") === "tab"
                || link.classList.contains("nav-group-item")) {
                link.setAttribute("aria-selected", isActive ? "true" : "false");
            }
        });

        // Toggle parent group triggers so the dropdown label highlights
        // when one of its children is the active tab.
        const activeGroup = TAB_TO_GROUP[name] || null;
        document.querySelectorAll(".nav-group-trigger").forEach((trig) => {
            const isActive = trig.dataset.group === activeGroup;
            trig.classList.toggle("active", isActive);
            // aria-current is the cleanest way to mark this; aria-selected
            // is reserved for tabs proper.
            if (isActive) {
                trig.setAttribute("aria-current", "true");
            } else {
                trig.removeAttribute("aria-current");
            }
        });

        // Toggle panels
        document.querySelectorAll(".tab-panel").forEach((panel) => {
            const isActive = panel.dataset.tab === name;
            if (isActive) {
                panel.removeAttribute("hidden");
                panel.classList.remove("active");
                void panel.offsetWidth;
                panel.classList.add("active");
                runRevealPass(panel);
            } else {
                panel.classList.remove("active");
                panel.setAttribute("hidden", "");
            }
        });

        // Close any open dropdown — keep the nav tidy after a click.
        closeAllDropdowns();

        window.scrollTo({ top: 0, behavior: "smooth" });

        if (focusInput && name === "chat") {
            const input = document.getElementById("chat-input");
            if (input) setTimeout(() => input.focus(), 80);
        }

        window.dispatchEvent(new Event("resize"));
    }

    function runRevealPass(panel) {
        panel.querySelectorAll(".reveal, .reveal-card").forEach((el) => {
            el.classList.remove("revealed");
            void el.offsetWidth;
            requestAnimationFrame(() => el.classList.add("revealed"));
        });
        panel.querySelectorAll(".stagger").forEach((group) => {
            const children = Array.from(group.children);
            children.forEach((child, i) => {
                child.classList.remove("revealed");
                child.style.transitionDelay = `${i * 60}ms`;
            });
            void group.offsetWidth;
            requestAnimationFrame(() => {
                children.forEach((child) => child.classList.add("revealed"));
            });
        });
    }

    // ── Dropdown menu helpers ─────────────────────────────────────
    function closeAllDropdowns() {
        document.querySelectorAll(".nav-group.is-open").forEach((g) => {
            g.classList.remove("is-open");
            const trig = g.querySelector(".nav-group-trigger");
            if (trig) trig.setAttribute("aria-expanded", "false");
        });
    }

    function toggleDropdown(group) {
        const wasOpen = group.classList.contains("is-open");
        closeAllDropdowns();
        if (!wasOpen) {
            group.classList.add("is-open");
            const trig = group.querySelector(".nav-group-trigger");
            if (trig) trig.setAttribute("aria-expanded", "true");
        }
    }

    // ── Wire up nav clicks ──
    document.addEventListener("click", (ev) => {
        // Dropdown trigger? Open its menu — do NOT route.
        const trigger = ev.target.closest(".nav-group-trigger");
        if (trigger) {
            ev.preventDefault();
            const group = trigger.closest(".nav-group");
            if (group) toggleDropdown(group);
            return;
        }

        // Tab link?  Route via hash.
        const link = ev.target.closest(".nav-link, .nav-brand");
        if (!link) {
            // Outside-click closes any open menu.
            if (!ev.target.closest(".nav-group")) closeAllDropdowns();
            return;
        }
        const tab = link.dataset.tab;
        if (!VALID_TABS.has(tab)) return;
        ev.preventDefault();
        if (location.hash !== `#${tab}`) {
            history.pushState(null, "", `#${tab}`);
            setActiveTab(tab, { focusInput: true });
        } else {
            setActiveTab(tab, { focusInput: true });
        }
    });

    // Close dropdowns on Escape
    document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") closeAllDropdowns();
    });

    // ── Browser back/forward ──
    window.addEventListener("hashchange", () => {
        setActiveTab(getHashTab());
    });

    // ── Boot ──
    document.addEventListener("DOMContentLoaded", () => {
        setActiveTab(getHashTab());
    });
})();
