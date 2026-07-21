"""Unit tests for `DiagnoseWorker` itself (not going through the MCP tool layer) --
verifies the hard-timeout kill-and-respawn behavior deterministically and fast,
without needing to reproduce a real 200+-second Oxigraph hang. Real callers always
use `DiagnoseWorker`'s defaults; `worker_loop`/`hard_timeout` are only overridable
for exactly this kind of test.
"""

from __future__ import annotations

import time

import pytest

from sparql_relax_mcp.server import DiagnoseWorker, _diagnose_worker_loop


def _hangs_forever(conn) -> None:
    """A worker loop that never replies -- stands in for a disconnected-BGP
    combination that never yields control back, without actually waiting on one."""
    conn.recv()
    time.sleep(3600)


def test_a_hung_call_is_killed_at_the_hard_timeout_not_the_real_delay():
    worker = DiagnoseWorker(worker_loop=_hangs_forever, hard_timeout=0.3)
    try:
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="hard timeout"):
            worker.call("some-dataset", "diagnose", "SELECT * WHERE { ?s ?p ?o }")
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, "should be killed at ~hard_timeout, not made to wait anywhere near the simulated hang"
    finally:
        worker.shutdown()


def test_the_worker_process_is_replaced_after_a_timeout():
    worker = DiagnoseWorker(worker_loop=_hangs_forever, hard_timeout=0.3)
    try:
        pid_before = worker._proc.pid
        with pytest.raises(RuntimeError):
            worker.call("some-dataset", "diagnose", "SELECT * WHERE { ?s ?p ?o }")
        pid_after = worker._proc.pid
        assert pid_after != pid_before, "a timed-out worker should be killed and replaced, not reused"
        assert worker._proc.is_alive()
    finally:
        worker.shutdown()


def test_invalidate_replaces_the_worker_process():
    worker = DiagnoseWorker()
    try:
        pid_before = worker._proc.pid
        worker.invalidate()
        assert worker._proc.pid != pid_before
        assert worker._proc.is_alive()
    finally:
        worker.shutdown()


def test_a_normal_call_through_the_real_worker_loop_works():
    """Sanity check with the actual `_diagnose_worker_loop` (not a test stand-in):
    a request for a dataset the worker has never heard of should come back as a
    clear error, not a hang or a crash -- proving the request/response plumbing
    itself (not just the timeout path) works end to end."""
    worker = DiagnoseWorker(worker_loop=_diagnose_worker_loop, hard_timeout=5.0)
    try:
        with pytest.raises(RuntimeError, match="no dataset named"):
            worker.call("nonexistent", "diagnose", "SELECT * WHERE { ?s ?p ?o }")
    finally:
        worker.shutdown()
