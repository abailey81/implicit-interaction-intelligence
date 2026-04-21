/* ============================================================================
 * MathJax v3 configuration for I³ documentation.
 *
 * Uses the CHTML renderer (shipped via the MathJax CDN in extra_javascript)
 * with inline-math and display-math delimiters compatible with the
 * pymdownx.arithmatex "generic" mode.
 * ============================================================================ */

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

/* Re-typeset math on instant-navigation page changes. */
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
