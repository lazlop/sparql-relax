# SPARQL-Relax

SPARQL-Relax is a toolkit for diagnosing and repairing broken SPARQL queries over RDF graphs. When a query returns no results but is expected to, SPARQL-Relax helps identify the "culprit" triple pattern or filter and suggests a "relaxed" version of the query by searching for actual paths in the knowledge graph.

## Repository Structure

- `sparql-relax-core/`: The Rust implementation of the diagnosis and relaxation logic.
- `sparql-relax-py/`: Python bindings for the Rust core, providing a high-level API for developers.
- `sparql-relax-mcp/`: An MCP (Model Context Protocol) server that enables AI agents to use the diagnosis and relaxation tools.
- `eval/`: Evaluation framework containing benchmarks and scripts (`run_eval.py`) to measure the effectiveness of the repair process on real-world datasets.

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
