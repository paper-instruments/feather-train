"""Execute one tool inside the agent container; emit JSON state to the host."""

import asyncio
import json
import signal
import sys
from pathlib import Path

from _file_worker import run_file_worker
from _files import FileToolError
from _process_output import format_process_result, format_task_list_entry
from process_worker import managed_groups, signal_groups, start_task, terminate
from save_transcript import save_transcript


def emit(value):
    payload = json.dumps(value, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.buffer.flush()


async def file_call(config, name, arguments):
    payload = json.dumps({**config, "arguments": arguments}).encode()
    returncode, stdout, stderr = await run_file_worker(name, payload)
    if returncode:
        raise RuntimeError(f"{name} failed: {stderr.decode(errors='replace').strip()}")
    return json.loads(stdout)


async def run(request):
    config, args = request["config"], request["arguments"]
    if request["name"] == "_save_transcript":
        path = Path(config["log_directory"]).parent / "transcript.jsonl"
        save_transcript(path, args["content"])
        emit({"done": True, "result": {"content": ""}})
        return
    if request["name"] != "bash":
        emit({"done": True, "result": await file_call(config, request["name"], args)})
        return

    published = None

    def publish(task):
        nonlocal published
        if task.state == "failed":
            return  # The exception below supplies the infrastructure failure.
        version = (task.state, task.output.total_bytes)
        if version == published:
            emit({"heartbeat": True})
            return
        published = version
        emit(
            {
                "done": task.state != "running",
                "result": {"content": format_process_result(task)},
                "list_entry": format_task_list_entry(task),
            }
        )

    task = await start_task(
        args,
        workspace=config["workspace"],
        grants=config["grants"],
        task_id=request["task_id"],
        log_file=Path(request["log_file"]),
        token=request["id"],
        on_update=publish,
    )
    if request["background"]:
        publish(task)
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM,
        lambda: task.monitor.cancel() if not task.terminating else None,
    )
    await task.monitor


async def main():
    if len(sys.argv) > 1:
        # Registry ownership lives outside this container. It can stop both the
        # supervisor and detached children even if either process has died.
        for token in sys.argv[1:]:
            supervisor = token + "-supervisor"
            signal_groups(await managed_groups(0, supervisor), signal.SIGTERM)
            # Give the supervisor time to reap stubborn children and publish the
            # final stopped result before falling back to independent cleanup.
            for _ in range(30):
                if not await managed_groups(0, supervisor):
                    break
                await asyncio.sleep(0.1)
            signal_groups(await managed_groups(0, supervisor), signal.SIGKILL)
            await terminate(0, token)
        return
    emit({"started": True})
    request = json.load(sys.stdin)
    try:
        await run(request)
    except FileToolError as exc:
        emit({"done": True, "result": {"content": str(exc), "error": exc.category}})
    except Exception as exc:
        emit({"done": True, "failure": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    asyncio.run(main())
