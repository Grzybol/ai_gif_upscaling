"""Bundle completed video-converter jobs into one downloadable archive."""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from midnight_upscale.utils import PipelineError, next_available_file


def create_batch_zip(
    jobs: list[tuple[Path, list[Path]]], output_dir: Path, *, overwrite: bool
) -> Path:
    """Place each source video's exported files below its own archive directory."""

    if not jobs:
        raise PipelineError("No completed videos to bundle.")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "video_converter_batch.zip"
    if destination.exists() and not overwrite:
        destination = next_available_file(destination)
    temporary = destination.with_name(destination.name + ".partial")
    used_names: set[str] = set()
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for source, outputs in jobs:
                base = re.sub(r"[^\w.-]+", "_", source.stem, flags=re.UNICODE).strip("._")
                base = base or "video"
                name = base
                suffix = 2
                while name.casefold() in used_names:
                    name = f"{base}_{suffix}"
                    suffix += 1
                used_names.add(name.casefold())
                for output in dict.fromkeys(outputs):
                    if output.is_dir():
                        for file in sorted(output.rglob("*")):
                            if file.is_file():
                                try:
                                    relative = file.resolve().relative_to(output_dir.resolve())
                                except ValueError:
                                    relative = Path(output.name) / file.relative_to(output)
                                archive.write(file, f"{name}/{relative.as_posix()}")
                    elif output.is_file():
                        try:
                            relative = output.resolve().relative_to(output_dir.resolve())
                        except ValueError:
                            relative = Path(output.name)
                        archive.write(output, f"{name}/{relative.as_posix()}")
        with ZipFile(temporary) as archive:
            bad_file = archive.testzip()
        if bad_file is not None:
            raise PipelineError(f"Batch ZIP validation failed at {bad_file}.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
