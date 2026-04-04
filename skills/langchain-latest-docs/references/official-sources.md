# Official LangChain Sources

Use these sources in priority order.

## Guides and Conceptual Docs

- Python OSS docs: `https://docs.langchain.com/oss/python/langchain/overview`
- JavaScript/TypeScript OSS docs: `https://docs.langchain.com/oss/javascript/langchain/overview`

Use these for:

- getting-started flows
- conceptual explanations
- recommended modern patterns
- migration direction and package positioning

## API Reference

- Python API reference root: `https://python.langchain.com/api_reference/`

Use this for:

- exact import paths
- class and function signatures
- method names and parameters
- checking whether an API still exists

## Version Verification

- PyPI package page: `https://pypi.org/project/langchain/`
- GitHub releases: `https://github.com/langchain-ai/langchain/releases`

Use these only when the user asks about:

- latest version
- release timing
- compatibility by version

Do not use release pages as the main source for behavior when the docs already cover it.

## Search Recipes

Choose the narrowest official-domain search that matches the task.

- Python concept question:
  `site:docs.langchain.com/oss/python/langchain <topic>`
- Python API lookup:
  `site:python.langchain.com/api_reference <symbol>`
- JavaScript/TypeScript concept question:
  `site:docs.langchain.com/oss/javascript/langchain <topic>`
- Latest version lookup:
  `site:pypi.org/project/langchain langchain`

## Answering Standard

- Prefer official docs over memory.
- Quote versions only after verifying them.
- Include the source URL when giving exact API guidance.
- If an answer depends on a package split, name the package explicitly.
