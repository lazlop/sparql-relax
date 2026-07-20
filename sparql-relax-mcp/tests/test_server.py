"""In-process tests for the MCP server: drives it through a real `ClientSession` over
in-memory transports (no subprocess, no stdio), exercising the same call_tool path a
real MCP client would use.
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from sparql_relax_mcp.server import _datasets, mcp

# Uses the Brick namespace (rather than an arbitrary made-up one) because diagnose's
# relaxation path search defaults to Brick/223P/RDFS/QUDT predicates only (see
# DEFAULT_RELAX_NAMESPACES) -- a fix outside those namespaces would never be found,
# which would make test_diagnose_explains_a_broken_query_and_suggests_a_fix below
# fail for a reason unrelated to what it's actually checking.
TTL = """
@prefix ex: <https://brickschema.org/schema/Brick#> .
ex:building223 ex:hasPart ex:zone1 .
ex:zone1 ex:hasSensor ex:sensor1 .
ex:sensor1 a ex:TempSensor .
ex:sensor2 a ex:TempSensor .
"""

WORKING_QUERY = "PREFIX ex: <https://brickschema.org/schema/Brick#> SELECT ?s WHERE { ?s a ex:TempSensor }"
BROKEN_QUERY = """
PREFIX ex: <https://brickschema.org/schema/Brick#>
SELECT ?sensor WHERE {
    ex:building223 ex:hasSensor ?sensor .
    ?sensor a ex:TempSensor .
}
"""


@pytest.fixture(autouse=True)
def _clear_datasets():
    """Datasets are process-global module state; reset between tests so they don't leak."""
    _datasets.clear()
    yield
    _datasets.clear()


def _result_json(call_tool_result) -> dict:
    assert not call_tool_result.isError, call_tool_result.content
    assert call_tool_result.structuredContent is not None
    return call_tool_result.structuredContent


@pytest.mark.asyncio
async def test_lists_all_four_tools():
    async with create_connected_server_and_client_session(mcp) as client:
        tools = (await client.list_tools()).tools
        assert {t.name for t in tools} == {"load_dataset", "list_datasets", "diagnose", "query"}


@pytest.mark.asyncio
async def test_load_then_list_datasets():
    async with create_connected_server_and_client_session(mcp) as client:
        loaded = _result_json(await client.call_tool("load_dataset", {"name": "b223", "data": TTL}))
        assert loaded == {"name": "b223", "format": "turtle", "triple_count": 4}

        listed = _result_json(await client.call_tool("list_datasets", {}))
        assert listed["result"] == [{"name": "b223", "format": "turtle", "triple_count": 4}]


@pytest.mark.asyncio
async def test_load_dataset_rejects_both_data_and_path():
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool("load_dataset", {"name": "x", "data": TTL, "path": "/tmp/nonexistent.ttl"})
        assert result.isError


@pytest.mark.asyncio
async def test_query_without_loading_dataset_first_is_a_clear_error():
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool("query", {"dataset": "missing", "query": "SELECT * WHERE { ?s ?p ?o }"})
        assert result.isError
        text = "".join(block.text for block in result.content if block.type == "text")
        assert "no dataset named 'missing'" in text
        assert "load_dataset" in text


@pytest.mark.asyncio
async def test_diagnose_reports_ok_on_a_working_query_then_query_fetches_full_results():
    async with create_connected_server_and_client_session(mcp) as client:
        await client.call_tool("load_dataset", {"name": "b223", "data": TTL})

        diagnosis = _result_json(await client.call_tool("diagnose", {"dataset": "b223", "query": WORKING_QUERY}))
        assert diagnosis["ok"] is True
        assert diagnosis["row_count"] == 2
        assert diagnosis["culprits"] == []
        assert diagnosis["filter_issues"] == []

        result = _result_json(await client.call_tool("query", {"dataset": "b223", "query": WORKING_QUERY}))
        assert result["form"] == "solutions"
        assert result["variables"] == ["s"]
        values = {row["s"]["value"] for row in result["rows"]}
        assert values == {"https://brickschema.org/schema/Brick#sensor1", "https://brickschema.org/schema/Brick#sensor2"}
        assert all(row["s"]["type"] == "uri" for row in result["rows"])


@pytest.mark.asyncio
async def test_diagnose_explains_a_broken_query_and_suggests_a_fix():
    async with create_connected_server_and_client_session(mcp) as client:
        await client.call_tool("load_dataset", {"name": "b223", "data": TTL})

        diagnosis = _result_json(await client.call_tool("diagnose", {"dataset": "b223", "query": BROKEN_QUERY}))
        assert diagnosis["ok"] is False
        assert diagnosis["row_count"] == 0
        assert len(diagnosis["culprits"]) == 1

        culprit = diagnosis["culprits"][0]
        assert culprit["triples"][0]["triple"] == "<https://brickschema.org/schema/Brick#building223> <https://brickschema.org/schema/Brick#hasSensor> ?sensor"
        assert culprit["fixed"] is True
        assert culprit["relaxed_query"] is not None
        assert culprit["row_count_with_fix"] > 0

        # The suggested relaxed_query should itself actually work via `query`.
        fixed = _result_json(await client.call_tool("query", {"dataset": "b223", "query": culprit["relaxed_query"]}))
        assert fixed["form"] == "solutions"
        assert len(fixed["rows"]) == culprit["row_count_with_fix"]


@pytest.mark.asyncio
async def test_query_row_limit_caps_solutions():
    async with create_connected_server_and_client_session(mcp) as client:
        await client.call_tool("load_dataset", {"name": "b223", "data": TTL})
        result = _result_json(await client.call_tool("query", {"dataset": "b223", "query": WORKING_QUERY, "row_limit": 1}))
        assert len(result["rows"]) == 1


@pytest.mark.asyncio
async def test_query_ask_and_construct_forms():
    async with create_connected_server_and_client_session(mcp) as client:
        await client.call_tool("load_dataset", {"name": "b223", "data": TTL})

        ask = _result_json(await client.call_tool("query", {"dataset": "b223", "query": "PREFIX ex: <https://brickschema.org/schema/Brick#> ASK { ex:sensor1 a ex:TempSensor }"}))
        assert ask == {"form": "boolean", "result": True}

        construct = _result_json(
            await client.call_tool(
                "query",
                {
                    "dataset": "b223",
                    "query": "PREFIX ex: <https://brickschema.org/schema/Brick#> CONSTRUCT { ?s a ex:Thing } WHERE { ?s a ex:TempSensor }",
                },
            )
        )
        assert construct["form"] == "graph"
        assert len(construct["triples"]) == 2


@pytest.mark.asyncio
async def test_diagnose_rejects_non_select_queries():
    async with create_connected_server_and_client_session(mcp) as client:
        await client.call_tool("load_dataset", {"name": "b223", "data": TTL})
        result = await client.call_tool("diagnose", {"dataset": "b223", "query": "ASK { ?s ?p ?o }"})
        assert result.isError
