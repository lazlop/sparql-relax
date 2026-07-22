# SPARQL-Relax

SPARQL-Relax is a toolkit for diagnosing broken SPARQL queries over RDF graphs. When a query returns no results but is expected to, SPARQL-Relax identifies the "culprit" triple pattern or filter — the core feature, and the one that's reliable enough to run on every query, including ones that already work. It also has an experimental relaxation mode that searches the graph for actual paths and proposes a "relaxed" fix; treat any suggested fix as a starting point to verify, not a guaranteed repair.

## Repository Structure

- `sparql-relax-core/`: The Rust implementation of the diagnosis logic, plus an experimental relaxation/repair search.
- `sparql-relax-py/`: Python bindings for the Rust core, providing a high-level API for developers.
- `sparql-relax-mcp/`: An MCP (Model Context Protocol) server that enables AI agents to use the diagnosis tool (and, experimentally, relaxation).
- `eval/`: Evaluation framework containing benchmarks and scripts (`run_eval.py`) to measure the effectiveness of diagnosis and relaxation on real-world datasets.

## Getting Started

### Use as a Python Library
If you want to integrate SPARQL-Relax into your own Python project, see the [Python Bindings README](./sparql-relax-py/README.md).
Quick start:
```bash
cd sparql-relax-py
maturin develop --release
```

Or add it to another project directly from GitHub with [`uv`](https://docs.astral.sh/uv/) (requires a Rust toolchain, since it builds the PyO3 extension from source):
```toml
[tool.uv.sources]
sparql-relax-rs = { git = "https://github.com/lazlop/sparql-relax", subdirectory = "sparql-relax-py" }
```
```bash
uv add sparql-relax-rs
```

### Use with AI Agents (MCP)
To use SPARQL-Relax as a tool for an AI agent (e.g., Claude), see the [MCP Server README](./sparql-relax-mcp/README.md).

Quick run with `uvx` (no clone needed, requires a Rust toolchain the first time):
```bash
uvx --from "git+https://github.com/lazlop/sparql-relax#subdirectory=sparql-relax-mcp" sparql-relax-mcp
```

> We may publish `sparql-relax-rs` / `sparql-relax-mcp` to PyPI (or ship prebuilt wheels) in the
> future so these installs don't require a local Rust toolchain. For now, installing from GitHub
> is the supported path.

### Learn by Example
We provide a Jupyter Notebook tutorial to get you started:
- [tutorial.ipynb](./tutorial.ipynb)

## Evaluation
The `eval/` directory contains tools to benchmark the system against generated queries and ground-truth results. You can run the evaluation script:
```bash
cd eval
python3 run_eval.py
```
