"""Create or overwrite a text file with revision checks."""

from pathlib import Path

from _files import (
    check_revision,
    load_text,
    main,
    require_current_read,
    require_write_target,
    validate_write_extension,
    write_text,
)


def write(payload):
    args = payload["arguments"]
    target = require_write_target(
        payload["workspace"],
        payload["read_grants"],
        args["file_path"],
        write_grants=payload["write_grants"],
        allow_missing=True,
    )
    display = target.display
    validate_write_extension(Path(display), display)
    if target.metadata is not None:
        expected = require_current_read(args["revisions"], display)
        _, encoding, actual = load_text(target.path, display, target.metadata, scope="writes text files")
        check_revision(display, expected, actual)
    else:
        encoding = "utf-8"
    # An exact file grant does not authorize creating its parent directories.
    create_parents = target.metadata is None and not target.exact_grant
    revision = write_text(target, args["content"], encoding, create_parents=create_parents)
    action = "Overwrote" if target.metadata is not None else "Created"
    return {
        "content": f"{action} {display}.",
        "metadata": {
            "path": display,
            "revision": revision,
            "encoding": encoding,
        },
    }


if __name__ == "__main__":
    main(write)
