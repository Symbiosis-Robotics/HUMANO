#!/usr/bin/env python3
"""Download up to two Objaverse-LVIS GLBs per category as backup assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "backup_assets" / "objaverse_lvis_2_per_category"
DEFAULT_REUSE_ROOT = REPO_ROOT / "data" / "resources" / "assets" / "objaverse_lvis"
HF_BASE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
ANNOTATIONS_URL = f"{HF_BASE}/lvis-annotations.json.gz"
OBJECT_PATHS_URL = f"{HF_BASE}/object-paths.json.gz"
CHUNK_SIZE = 8 * 1024 * 1024
UNKNOWN_DOWNLOAD_RESERVATION = 1024**3


class DiskSpaceGuard(RuntimeError):
    """Raised when continuing a download would violate the free-space floor."""


@dataclass(frozen=True)
class Sample:
    category: str
    directory: str
    rank: int
    uid: str
    source_path: str


@dataclass
class Result:
    category: str
    directory: str
    rank: int
    uid: str
    source_path: str
    destination: str
    status: str
    bytes: int = 0
    reused_from: str | None = None
    reason: str | None = None


class DiskBudget:
    def __init__(self, root: Path, min_free_bytes: int) -> None:
        self.root = root
        self.min_free_bytes = min_free_bytes
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._reserved = 0

    def reserve(self, requested: int) -> int:
        reservation = requested if requested > 0 else UNKNOWN_DOWNLOAD_RESERVATION
        with self._lock:
            free = shutil.disk_usage(self.root).free
            if free - self._reserved - reservation < self.min_free_bytes:
                self.stop.set()
                raise DiskSpaceGuard(
                    f"disk guard: free={free}, reserved={self._reserved}, "
                    f"next={reservation}, minimum_free={self.min_free_bytes}"
                )
            self._reserved += reservation
        return reservation

    def release(self, reservation: int) -> None:
        with self._lock:
            self._reserved -= reservation

    def check_floor(self) -> None:
        if self.stop.is_set() or shutil.disk_usage(self.root).free < self.min_free_bytes:
            self.stop.set()
            raise DiskSpaceGuard(
                f"disk guard reached minimum free space: {self.min_free_bytes} bytes"
            )


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return cleaned or "unnamed"


def valid_file(path: Path, expected_magic: bytes) -> bool:
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with path.open("rb") as stream:
        return stream.read(len(expected_magic)) == expected_magic


def valid_glb(path: Path) -> bool:
    return valid_file(path, b"glTF")


def seed_metadata(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if source.is_file() and source.stat().st_size > 0:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def download_file(
    url: str,
    destination: Path,
    retries: int,
    budget: DiskBudget,
    *,
    expected_magic: bytes = b"glTF",
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(retries):
        reservation = 0
        try:
            budget.check_floor()
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"User-Agent": "HUMANO-Objaverse-LVIS-Backup/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                partial_response = getattr(response, "status", None) == 206
                if offset and not partial_response:
                    offset = 0
                content_length = int(response.headers.get("Content-Length") or 0)
                reservation = budget.reserve(content_length)
                mode = "ab" if offset and partial_response else "wb"
                with partial.open(mode) as output:
                    while True:
                        budget.check_floor()
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)

            if not valid_file(partial, expected_magic):
                raise RuntimeError(f"downloaded file has invalid magic: {destination.name}")
            partial.replace(destination)
            return destination.stat().st_size
        except DiskSpaceGuard:
            raise
        except Exception as exc:  # Network failures retain the partial file for resume.
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        finally:
            if reservation:
                budget.release(reservation)
    raise RuntimeError(str(last_error))


def load_gzip_json(
    url: str,
    destination: Path,
    source: Path,
    retries: int,
    budget: DiskBudget,
) -> dict:
    seed_metadata(source, destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        download_file(url, destination, retries, budget, expected_magic=b"\x1f\x8b")
    with gzip.open(destination, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def build_samples(
    annotations: dict[str, list[str]],
    object_paths: dict[str, str],
    samples_per_category: int,
) -> tuple[list[Sample], list[dict]]:
    samples: list[Sample] = []
    shortfalls: list[dict] = []
    used_directories: set[str] = set()
    for category in sorted(annotations):
        directory = safe_name(category)
        if directory in used_directories:
            raise ValueError(f"sanitized category collision: {category!r} -> {directory!r}")
        used_directories.add(directory)

        selected: list[str] = []
        for uid in annotations[category]:
            if uid in selected or uid not in object_paths:
                continue
            selected.append(uid)
            if len(selected) == samples_per_category:
                break
        for rank, uid in enumerate(selected, start=1):
            samples.append(Sample(category, directory, rank, uid, object_paths[uid]))
        if len(selected) < samples_per_category:
            shortfalls.append(
                {
                    "category": category,
                    "directory": directory,
                    "requested": samples_per_category,
                    "available": len(selected),
                }
            )
    return samples, shortfalls


def destination_for(download_root: Path, sample: Sample) -> Path:
    return download_root / sample.directory / f"{sample.uid}.glb"


def reuse_file(source: Path, destination: Path, budget: DiskBudget) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if valid_glb(destination):
            return "skipped_existing", destination.stat().st_size
        raise RuntimeError(f"invalid destination already exists: {destination}")

    reservation = budget.reserve(source.stat().st_size)
    try:
        budget.check_floor()
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        budget.release(reservation)
    return "copied_existing", destination.stat().st_size


def download_sample(
    sample: Sample,
    download_root: Path,
    reuse_root: Path | None,
    retries: int,
    budget: DiskBudget,
) -> Result:
    destination = destination_for(download_root, sample)
    base = {
        **asdict(sample),
        "destination": str(destination),
    }
    if valid_glb(destination):
        return Result(**base, status="skipped_existing", bytes=destination.stat().st_size)
    if destination.exists() and destination.suffix != ".part":
        return Result(**base, status="failed", reason="invalid existing destination")
    if budget.stop.is_set():
        return Result(**base, status="not_attempted_disk_guard", reason="disk guard already active")

    if reuse_root is not None:
        reusable = reuse_root / sample.directory / f"{sample.uid}.glb"
        if valid_glb(reusable):
            try:
                status, size = reuse_file(reusable, destination, budget)
                return Result(
                    **base,
                    status=status,
                    bytes=size,
                    reused_from=str(reusable),
                )
            except DiskSpaceGuard as exc:
                return Result(**base, status="blocked_disk_space", reason=str(exc))
            except Exception as exc:
                return Result(**base, status="failed", reason=f"reuse failed: {exc}")

    try:
        size = download_file(f"{HF_BASE}/{sample.source_path}", destination, retries, budget)
        return Result(**base, status="downloaded", bytes=size)
    except DiskSpaceGuard as exc:
        return Result(**base, status="blocked_disk_space", reason=str(exc))
    except Exception as exc:
        return Result(**base, status="failed", reason=str(exc))


def summarize(results: list[Result]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return dict(sorted(summary.items()))


def write_manifest(
    path: Path,
    *,
    download_root: Path,
    samples_per_category: int,
    min_free_bytes: int,
    samples: list[Sample],
    shortfalls: list[dict],
    results: list[Result],
    started_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "allenai/objaverse",
        "subset": "Objaverse-LVIS",
        "selection": f"first {samples_per_category} distinct official UIDs per LVIS category",
        "download_root": str(download_root.resolve()),
        "samples_per_category": samples_per_category,
        "selected_category_count": len({sample.category for sample in samples}),
        "selected_asset_count": len(samples),
        "category_shortfalls": shortfalls,
        "minimum_free_bytes": min_free_bytes,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize(results),
        "samples": [asdict(sample) for sample in samples],
        "results": [asdict(result) for result in results],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reuse-root", type=Path, default=DEFAULT_REUSE_ROOT)
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit", type=int, help="Limit categories for a smoke test.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_category < 1 or args.workers < 1 or args.min_free_gib < 0:
        raise ValueError("samples, workers, and free-space settings must be positive")

    root = args.root.resolve()
    download_root = root / "raw_glb"
    metadata_root = root / "_metadata"
    manifest = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    min_free_bytes = int(args.min_free_gib * 1024**3)
    budget = DiskBudget(root, min_free_bytes)
    reuse_root = None if args.no_reuse_existing else args.reuse_root.resolve()
    reuse_metadata = (
        reuse_root / "_metadata"
        if reuse_root is not None
        else root / "_no_reuse_metadata"
    )

    annotations = load_gzip_json(
        ANNOTATIONS_URL,
        metadata_root / "lvis-annotations.json.gz",
        reuse_metadata / "lvis-annotations.json.gz",
        args.retries,
        budget,
    )
    object_paths = load_gzip_json(
        OBJECT_PATHS_URL,
        metadata_root / "object-paths.json.gz",
        reuse_metadata / "object-paths.json.gz",
        args.retries,
        budget,
    )
    samples, shortfalls = build_samples(annotations, object_paths, args.samples_per_category)

    if args.category:
        requested = set(args.category)
        known = {sample.category for sample in samples}
        missing = requested - known
        if missing:
            raise ValueError(f"unknown LVIS categories: {sorted(missing)}")
        samples = [sample for sample in samples if sample.category in requested]
        shortfalls = [item for item in shortfalls if item["category"] in requested]
    if args.limit is not None:
        categories = sorted({sample.category for sample in samples})[: args.limit]
        selected = set(categories)
        samples = [sample for sample in samples if sample.category in selected]
        shortfalls = [item for item in shortfalls if item["category"] in selected]

    print(f"categories: {len({sample.category for sample in samples})}")
    print(f"assets: {len(samples)}")
    print(f"category shortfalls: {len(shortfalls)}")
    print(f"root: {root}")
    if args.dry_run:
        for sample in samples:
            print(f"{sample.category}\t{sample.rank}\t{sample.uid}\t{sample.source_path}")
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[Result] = []
    sample_iterator = iter(samples)
    pending: dict[concurrent.futures.Future[Result], Sample] = {}
    last_manifest_write = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        def submit_next() -> bool:
            try:
                sample = next(sample_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                download_sample,
                sample,
                download_root,
                reuse_root,
                args.retries,
                budget,
            )
            pending[future] = sample
            return True

        for _ in range(args.workers):
            if not submit_next():
                break

        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                pending.pop(future)
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results)}/{len(samples)}] {result.status}: "
                    f"{result.directory}[{result.rank}] ({result.bytes} bytes)",
                    flush=True,
                )
                if result.reason:
                    print(f"  reason: {result.reason}", flush=True)
                now = time.monotonic()
                if len(results) % 25 == 0 or now - last_manifest_write >= 10.0:
                    write_manifest(
                        manifest,
                        download_root=download_root,
                        samples_per_category=args.samples_per_category,
                        min_free_bytes=min_free_bytes,
                        samples=samples,
                        shortfalls=shortfalls,
                        results=results,
                        started_at=started_at,
                    )
                    last_manifest_write = now
                if not budget.stop.is_set():
                    submit_next()

    if budget.stop.is_set():
        for sample in sample_iterator:
            results.append(
                Result(
                    **asdict(sample),
                    destination=str(destination_for(download_root, sample)),
                    status="not_attempted_disk_guard",
                    reason="minimum free-space guard was reached",
                )
            )

    write_manifest(
        manifest,
        download_root=download_root,
        samples_per_category=args.samples_per_category,
        min_free_bytes=min_free_bytes,
        samples=samples,
        shortfalls=shortfalls,
        results=results,
        started_at=started_at,
    )
    summary = summarize(results)
    print(f"summary: {json.dumps(summary, sort_keys=True)}")
    print(f"manifest: {manifest}")
    return 1 if summary.get("failed", 0) or budget.stop.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
