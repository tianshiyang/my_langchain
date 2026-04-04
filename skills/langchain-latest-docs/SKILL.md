---
name: langchain-latest-docs
description: Verify LangChain answers against the latest official documentation before responding. Use when the task involves LangChain Python or JavaScript/TypeScript APIs, LCEL, chains, agents, tools, prompts, callbacks, streaming, output parsers, retrievers, document loaders, LangGraph interplay, package splits, migrations, or version-specific behavior. Prefer official docs and API reference over memory.
---

# LangChain Latest Docs

Use this skill whenever the user is asking about LangChain behavior that may have changed, needs exact API usage, or mentions a versioned migration path.

## Trigger Conditions

Use this skill when the request includes any of the following:

- `LangChain`, `langchain`, `langchain-core`, `langchain-openai`, `langchain-community`
- `LCEL`, prompt templates, runnables, output parsers, tools, agents, callbacks, streaming
- retrievers, vector stores, loaders, text splitters, memory, middleware
- migration questions such as `0.x -> 1.x`, package rename/split questions, or "latest LangChain"
- LangChain plus `LangGraph`, especially when the boundary between the two matters

## Workflow

### 1. Identify the target surface

Before answering, determine:

- language: Python or JavaScript/TypeScript
- question type: conceptual guide, migration guidance, or API-level reference
- package family: `langchain`, `langchain-core`, `langchain-openai`, `langgraph`, or another related package

If the user did not specify a language, infer it from the repo or the code shown. If that is still unclear, state the assumption you are making.

### 2. Verify with official sources first

Do not answer from memory when current docs are available. Prefer official sources in this order:

1. LangChain OSS docs for guides and conceptual usage
2. LangChain API reference for exact classes, methods, arguments, and module locations
3. Official release/version sources only when the user asks about "latest" or version compatibility

Read [official-sources.md](./references/official-sources.md) for the exact entry points and search recipes.

### 3. Resolve version-sensitive details

If the question depends on recency:

- verify the current package version first
- verify that the referenced docs page matches the current API shape
- include a concrete version or access date in the answer when helpful

If docs and memory conflict, the docs win.

### 4. Answer with citations and clear boundaries

When responding:

- cite the official page URLs you used
- prefer short, direct code examples that match the docs
- call out package splits or renamed imports when relevant
- explicitly say when something could not be verified from the official docs

Never present guessed imports, outdated class names, or pre-`1.x` patterns as current truth.

## Search Rules

- Search official LangChain domains first; avoid third-party blogs unless the user explicitly asks for them.
- For Python questions, prioritize Python docs and Python API reference.
- For JavaScript/TypeScript questions, prioritize JavaScript docs.
- For version checks, prefer official package/release sources over community summaries.

Good search pattern examples:

- `site:docs.langchain.com/oss/python/langchain <topic>`
- `site:python.langchain.com/api_reference <class or function>`
- `site:docs.langchain.com/oss/javascript/langchain <topic>`
- `site:pypi.org/project/langchain latest langchain`

## Output Style

- Lead with the current, verified answer.
- If the user's wording suggests an outdated API, gently correct it and show the modern equivalent.
- Keep migration notes short and practical.
- If multiple packages are involved, name each one explicitly so the user can install/import the right dependency.

## Common Pitfalls

- confusing `LangChain` and `LangGraph`
- using pre-`1.0` chain helpers when the current docs recommend runnables or newer patterns
- importing integrations from the wrong package after package splits
- answering from stale memory when the docs have moved or been reorganized

If uncertainty remains after checking the official docs, say that clearly and avoid overclaiming.
