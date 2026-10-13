"""Optional benchmark evidence capture before the agent container is removed."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from feather_train.sandbox.capture_files import MAX_ARCHIVE_BYTES
from feather_train.sandbox.lifecycle import SandboxIO

if TYPE_CHECKING:
    from feather_train.harness import FeatherHarnessConfig
    from feather_train.sandbox.lifecycle import ComposeSandbox
    from feather_train.tasks.taskset import FeatherTaskData

logger = logging.getLogger(__name__)
CAPTURE_TIMEOUT_SECONDS = 180


async def capture_deliverables(
    world: ComposeSandbox,
    task: FeatherTaskData,
    *,
    trace_id: str,
    final_answer: str,
    model: str,
    sampling: dict,
    config: FeatherHarnessConfig,
) -> dict:
    """Return a receipt; capture failures never invalidate benchmark grading."""
    relative = Path(task.eval_run_id) / task.task_id / trace_id
    receipt = {
        "schema_version": 1,
        "status": "failed",
        "run_id": task.eval_run_id,
        "task_id": task.task_id,
        "trace_id": trace_id,
        "world_id": task.world_id,
        "hf_dataset": task.hf_dataset,
        "hf_revision": task.hf_revision,
        "sandbox_snapshot_id": task.e2b_snapshot_id,
        "model": model,
        "profile": config.profile,
        "sampling": sampling,
        "roots": list(config.capture_paths),
        "captured_at": datetime.now(UTC).isoformat(),
        "storage": "modal_volume" if config.capture_volume_name else "local",
        "volume": config.capture_volume_name,
        "path": str(relative),
        "output_dir": config.capture_output_dir,
        "code_revision": os.environ.get("TRAIN_DEPLOYMENT_REVISION"),
    }
    try:
        async with asyncio.timeout(CAPTURE_TIMEOUT_SECONDS):
            destination = Path(config.capture_output_dir) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".capture-", dir=destination.parent) as directory:
                temporary = Path(directory)
                inventory = await _download_archive(world, config.capture_paths, temporary / "files.tar.gz")
                receipt.update(
                    inventory,
                    archive="files.tar.gz",
                    final_response_sha256=hashlib.sha256(final_answer.encode()).hexdigest(),
                    status="captured",
                )
                await asyncio.to_thread(_publish, destination, temporary, final_answer, receipt)
            if config.capture_volume_name:
                import modal

                await modal.Volume.from_name(config.capture_volume_name).commit.aio()
            receipt["status"] = "complete"
    except Exception as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        logger.warning("Deliverable capture failed task=%s trace=%s", task.task_id, trace_id, exc_info=True)
    return receipt


async def _download_archive(world: ComposeSandbox, roots: tuple[str, ...], destination: Path) -> dict:
    token = uuid.uuid4().hex
    container_path = f"/tmp/feather-deliverables-{token}.tar.gz"
    host_path = f"/tmp/feather-deliverables-host-{token}.tar.gz"
    agent = SandboxIO(world.sandbox, container="agent")
    host = SandboxIO(world.sandbox)
    script = Path(__file__).with_name("sandbox").joinpath("capture_files.py").read_text()
    try:
        result = await agent.exec("python3", "-I", "-c", script, json.dumps(roots), container_path, timeout=120)
        if not result.ok():
            raise RuntimeError("Agent filesystem archiving failed")
        inventory = json.loads(result.stdout)
        copied = await host.exec("docker", "cp", f"agent:{container_path}", host_path, timeout=30)
        if not copied.ok():
            raise RuntimeError("Deliverable archive copy failed")
        size = 0
        digest = hashlib.sha256()
        async with await world.sandbox.files.read(
            host_path, format="stream", user="root", request_timeout=60
        ) as stream:
            with destination.open("xb") as output:
                async for chunk in stream:
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Deliverable archive download limit exceeded")
                    digest.update(chunk)
                    await asyncio.to_thread(output.write, chunk)
                await asyncio.to_thread(output.flush)
                await asyncio.to_thread(os.fsync, output.fileno())
        return {**inventory, "archive_bytes": size, "archive_sha256": digest.hexdigest()}
    finally:
        for io, path in ((agent, container_path), (host, host_path)):
            try:
                await io.exec("rm", "-f", path, timeout=10)
            except Exception:
                logger.warning("Could not clean up deliverable capture temporary file", exc_info=True)


def _publish(destination: Path, temporary: Path, final_answer: str, receipt: dict) -> None:
    for name, content in (
        ("final_response.txt", final_answer.encode("utf-8")),
        ("manifest.json", (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode()),
    ):
        with (temporary / name).open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    temporary.rename(destination)
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
