# Feather Eval

Evaluate Feather on Harbor tasks using an OpenAI-compatible model endpoint. The
adapter installs Feather into the task environment and runs its knowledge-work
tools and skills; Harbor manages the sandbox and grading.

- **Use existing tasks:** run Feather against compatible Harbor task directories
  with the task image supplying the Python and Office dependencies its scripts need.
- **Compare models:** point the same harness at different model endpoints.
- **Inspect trajectories:** keep Feather's native execution trace and export ATIF,
  with compaction contexts and summaries preserved.

## First run

From the workspace root, with Docker running and `uv` installed:

```bash
uv sync --package feather-eval
uv build --package feather-harness --wheel --out-dir dist

# Set OPENAI_API_KEY in your environment; it is forwarded to the agent process.
uv run --package feather-eval harbor run \
  -p feather-eval/tasks/file-smoke \
  -a feather_eval.harbor:FeatherHarborAgent \
  -m YOUR_MODEL \
  --ak "wheel=$PWD/dist/feather_harness-0.1.0-py3-none-any.whl" \
  --ak base_url=https://YOUR_PROVIDER/v1 \
  --ak max_turns=12 \
  --jobs-dir jobs -n 1
```

Replace `YOUR_MODEL` and `YOUR_PROVIDER` with your provider's model identifier and endpoint.

The bundled task asks Feather to summarize a CSV using file and shell tools. Its
image includes Office dependencies as a starting point for other tasks; grading
checks the CSV summary. To check the task setup with its supplied oracle solution:

```bash
uv run --package feather-eval harbor run \
  -p feather-eval/tasks/file-smoke -a oracle \
  --jobs-dir jobs -n 1
```

To evaluate another task, change `-p` to its Harbor task or dataset directory.
Feather uses the image's working directory by default; set `--ak workspace=...`
to override it. Rebuild the Harness wheel after changing the agent.

Harbor explicitly defaults to `profile=prod`, which uses professional production
prompting and guidance. Use `--ak profile=train` to reproduce the minimal training
profile instead.

For a vision-capable endpoint, pass `--ak enable_image_tools=true` (or set
`kwargs.enable_image_tools: true` in the Harbor agent config). This enables native
image reads through `read` for PNG/JPEG/GIF/WebP files in both profiles, plus image
results from MCP tools. Render document pages to images using the task's document
tools before inspecting them.
Images are retained in native traces and ATIF; the default remains text-only.

## Task environment

The adapter targets Harbor 0.22.0 and Linux task images. Setup needs root access,
a supported package manager, and network access to install Python and dependencies.
Feather runs in an isolated Python 3.12 environment. The adapter installs the bare
Harness, Bash, ripgrep, curl, and certificates. It preserves the task's `PATH` and
leaves its Python packages unchanged.

### Prepare the task image

Skills use ordinary `python`. The image should provide that command and install
its task dependencies into the same interpreter. For a virtualenv, put its `bin`
directory on `PATH`; on Ubuntu images with only `python3`, install `python-is-python3`.

For Office skills, add these two `RUN` statements to the task Dockerfile **after
its existing dependency installation**. The first installs Poppler for PDF tools;
the second replaces upstream Office packages with Paper, installs the remaining
Office dependencies, and checks the installation. This example assumes a
Debian-based image with `python` and pip available.

```dockerfile
# Existing task dependencies, such as Pandas, are installed above this step.
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip uninstall -y openpyxl python-docx python-pptx \
    && python -m pip install --no-cache-dir --force-reinstall \
       paper-docx==0.2.0 paper-pptx==0.2.0 paper-xlsx==0.2.1 \
       'reportlab>=4,<5' 'pypdf>=6,<7' 'pdfplumber>=0.11,<0.12' \
       'pdf2image>=1.17,<2' 'Pillow>=10.3,<13' \
    && paper-docx-doctor && paper-pptx-doctor && paper-xlsx-doctor
```

These dependencies match the Harness [`office` extra](../feather-harness/pyproject.toml).
Use a virtualenv for distro-managed Python, and remove distro-installed Office
packages with the system package manager before this step. Add LibreOffice or OCR
binaries when the task needs them; see the Harness
[system dependency commands](../feather-harness/README.md#install-and-run).

Graders sharing the task's Python environment also use Paper.

HTTP/SSE and stdio MCP declarations are supported. Harbor task skills are copied
into the workspace after bundled skills. Web tools are not currently exposed by
the adapter.

## Results and Harbor compatibility

Each run retains Feather's native `run.json` and exports ATIF for Harbor trajectory
tooling and task graders. `atif_mode` is a Feather adapter option; both modes use
Harbor's ATIF schema:

- **`segments` (default):** linked ATIF files preserve context windows and compaction
  summaries.
- **`grader`:** one `trajectory.json` contains the chronological actions and responses,
  excluding summaries and duplicated context. Use `--ak atif_mode=grader` for judges
  that read only that file and do not follow continuation links.

The agent log directory also contains `harness-condition.json`: the resolved
profile, effective image setting, and SHA-256 of the wheel supplied for
installation. Harbor context metadata includes the same
condition under `feather`, including when the CLI fails before producing a result.
The condition file is written before installation commands or model calls; it
does not imply those succeeded. It excludes credentials, MCP arguments, and
environment values. The hash distinguishes builds with the same package version;
retain the wheel itself if the exact build needs to be reproduced later.

To convert a saved Feather run separately, from the workspace root:

```bash
uv run --package feather-eval feather-atif run.json --output-dir atif --session-id my-run
```

Add `--grader` for a single chronological transcript.

## Development

From the workspace root:

```bash
uv run --package feather-eval --extra dev pytest -c feather-eval/pyproject.toml feather-eval/tests -q
uv run --package feather-eval --extra dev ruff check feather-eval/src feather-eval/tests
```

Local tests require Bash and use no cloud credentials or Docker. The `harbor run`
example above exercises installation and execution in a real task container.

## License

[MIT](LICENSE).
