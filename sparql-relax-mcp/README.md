# sparql-relax-mcp

An [MCP](https://modelcontextprotocol.io) server that lets AI agents load RDF graphs, run SPARQL
queries against them, and diagnose queries that return nothing (or less than expected) — built on
top of [`sparql_relax`](../sparql-relax-py).

## Tools

- **`load_dataset(name, data=None, path=None, format="turtle")`** — load RDF text (or a local
  file) into memory under `name`. Replaces any dataset already loaded under that name.
- **`list_datasets()`** — list loaded datasets with their format and triple count.
- **`diagnose(dataset, query)`** — run a SPARQL `SELECT` query and diagnose it. Cheap even when
  the query already works (`ok: true`); when it doesn't, explains which triple pattern or
  `FILTER` is broken and, when a real fix exists in the graph, suggests a corrected query.
- **`query(dataset, query, row_limit=1000)`** — run any SPARQL query form (`SELECT`, `ASK`,
  `CONSTRUCT`, `DESCRIBE`) and return the actual results.

**Intended workflow:** call `diagnose` on every new query before trusting its result — it's
nearly free when the query works and tells you exactly what's wrong when it doesn't. Only call
`query` once `diagnose` confirms rows come back (or directly, for `ASK`/`CONSTRUCT`/`DESCRIBE`,
which `diagnose` doesn't support).

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and a Rust toolchain (to build the `sparql-relax-rs`
extension the first time).

### Option A: run straight from GitHub (no clone)

```sh
uvx --from "git+https://github.com/lazlop/sparql-relax#subdirectory=sparql-relax-mcp" sparql-relax-mcp
```

This is the easiest way for collaborators to get the server without checking out the repo. `uv`
clones it, resolves the local `sparql-relax-rs` dependency, and builds the extension for you
(cached after the first run).

### Option B: from a local clone

```sh
uv sync
```

### Register with Claude Code

Project-scoped (add to `.mcp.json` in the repo root, or run from anywhere with `--directory`):

```sh
claude mcp add sparql-relax -- uv --directory /absolute/path/to/sparql-relax-mcp run sparql-relax-mcp
```

or by hand, in `.mcp.json`, using the GitHub install directly (no local path needed):

```json
{
  "mcpServers": {
    "sparql-relax": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/lazlop/sparql-relax#subdirectory=sparql-relax-mcp", "sparql-relax-mcp"]
    }
  }
}
```

or pointed at a local clone instead:

```json
{
  "mcpServers": {
    "sparql-relax": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/sparql-relax-mcp", "run", "sparql-relax-mcp"]
    }
  }
}
```

> We may publish `sparql-relax-rs` / `sparql-relax-mcp` to PyPI (or ship prebuilt wheels) in the
> future so this doesn't require a local Rust toolchain. For now, installing from GitHub is the
> supported path.

### Register with Claude Desktop

Add the same block to `claude_desktop_config.json` (Settings → Developer → Edit Config).

### Run directly (for testing)

```sh
uv run sparql-relax-mcp
```

Talks stdio — it will sit waiting for MCP protocol messages on stdin, not print a prompt. Use the
[MCP Inspector](https://modelcontextprotocol.io/legacy/tools/inspector) to poke at it manually:

```sh
npx @modelcontextprotocol/inspector uv run sparql-relax-mcp
```

## Development

```sh
uv sync --group dev
uv run pytest
```

`sparql-relax-rs` is installed editable from `../sparql-relax-py` (see `[tool.uv.sources]` in
`pyproject.toml`) — changes to the Rust core need `maturin develop --release` re-run in
`sparql-relax-py` (or `uv sync` here again) to take effect.
