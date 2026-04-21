/**
 * commitlint configuration for Implicit Interaction Intelligence (I3).
 *
 * Enforces Conventional Commits with a curated list of scopes that map 1:1
 * with the project's top-level modules so that release-please, the PR title
 * linter, and the changelog categorisation stay aligned.
 *
 * Docs: https://commitlint.js.org/
 */
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',
        'fix',
        'perf',
        'refactor',
        'docs',
        'test',
        'build',
        'ci',
        'chore',
        'revert',
        'security',
        'style',
        'deps',
      ],
    ],
    'scope-enum': [
      2,
      'always',
      [
        // Core algorithmic modules.
        'perception',
        'encoder',
        'user_model',
        'adaptation',
        'router',
        'slm',
        'cloud',
        'diary',
        'privacy',
        'profiling',
        'pipeline',

        // Application surfaces.
        'server',
        'web',

        // Lifecycle / tooling.
        'training',
        'demo',
        'tests',

        // Infra / meta.
        'ci',
        'deps',
        'docs',
        'docker',
        'k8s',
        'helm',
      ],
    ],
    'scope-empty': [1, 'never'],
    'scope-case': [2, 'always', 'kebab-case', 'snake_case'],
    'subject-case': [
      2,
      'never',
      ['sentence-case', 'start-case', 'pascal-case', 'upper-case'],
    ],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'subject-max-length': [2, 'always', 100],
    'header-max-length': [2, 'always', 120],
    'body-leading-blank': [2, 'always'],
    'body-max-line-length': [1, 'always', 120],
    'footer-leading-blank': [2, 'always'],
    'footer-max-line-length': [1, 'always', 120],
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],
  },
  ignores: [
    (message) => /^Merge /.test(message),
    (message) => /^Revert /.test(message),
    (message) => /^chore\(release\):/.test(message),
    (message) => /^Release /.test(message),
  ],
  helpUrl:
    'https://github.com/abailey81/i3/blob/main/CONTRIBUTING.md#commit-message-format',
};
