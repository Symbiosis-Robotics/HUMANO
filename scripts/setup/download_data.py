import io
import os
import shutil
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from urllib.request import urlopen

import typer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from humano.utils.paths import canonical_resource_path  # noqa: E402
from humano.utils.resource_layers import relocate_resource_layer  # noqa: E402

load_dotenv()

DEFAULT_REPO_ID = "USC-PSI-Lab/SIMPLE"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_LOCAL_DIR = "."
BRAINCO_REVISION = "da568324b86b74538dd1e08479103faa90dd9647"

CORE_ZIPS = (
    "vMaterials_2.zip",
    "assets_graspnet.zip",
    "assets_objects.zip",
    "assets_articulated.zip",
    "robots_g1.zip",
    "robots_g1_sonic.zip",
)

DEFAULT_SCENE_ZIPS = (
    "scenes_hssd_102344049.zip",
    "scenes_hssd_102344280.zip",
    "scenes_hssd_107734119_175999932.zip",
)


def _hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _unique_sorted(items: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(items))


def _download_file(
    filename: str,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
) -> Path:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        repo_type=repo_type,
        token=_hf_token(),
    )
    return _install_downloaded_resource(local_dir, filename)


def _install_downloaded_resource(local_dir: Path, filename: str) -> Path:
    relative = Path(filename)
    downloaded = local_dir / relative
    target = local_dir / canonical_resource_path(relative)
    if target != downloaded:
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)
        parent = downloaded.parent
        while parent != local_dir and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    if target != downloaded or relative.parts[:1] == ("scenes",):
        relocate_resource_layer(target)
    return target


def _extract_zip(zip_path: Path, local_dir: Path, keep_zip: bool) -> None:
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid resource archive path: {member.filename}")
            target = local_dir / canonical_resource_path(relative)
            if member.is_dir():
                # Legacy grouping directories may have no canonical equivalent.
                if relative.parts not in {
                    ("assets",),
                    ("assets_simplified_x10",),
                    ("scenes", "objects"),
                }:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if target != local_dir / relative or relative.parts[:1] == ("scenes",):
                relocate_resource_layer(target)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)


def _select_default_zips(
    repo_files: list[str],
    include_scenes: bool,
    all_scenes: bool,
) -> list[str]:
    zips: list[str] = [name for name in CORE_ZIPS if name in repo_files]
    if include_scenes:
        if all_scenes:
            zips.extend(
                name
                for name in repo_files
                if name.startswith("scenes_hssd_") and name.endswith(".zip")
            )
        else:
            zips.extend(name for name in DEFAULT_SCENE_ZIPS if name in repo_files)
    return _unique_sorted(zips)


def _download_brainco(local_dir: Path, dry_run: bool) -> None:
    """Install the fixed official Revo2 mechanical source, without other assets."""
    destination = local_dir / "robots/brainco/revo2"
    url = f"https://codeload.github.com/BrainCoTech/brainco-description/zip/{BRAINCO_REVISION}"
    print(f"BrainCo Revo2 {BRAINCO_REVISION}: {url} -> {destination}")
    if dry_run:
        return
    prefix = f"brainco-description-{BRAINCO_REVISION}/revo2_system/"
    with urlopen(url, timeout=60) as response:
        archive_bytes = response.read()
    installed = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = Path(member.filename.removeprefix(prefix))
            if relative.parts[0] not in {"meshes", "urdf", "README.md"}:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            installed += 1
    if not installed:
        raise RuntimeError(f"No Revo2 source assets found in {url}")
    print(f"Installed {installed} official Revo2 files under {destination}")


def download_from_huggingface(
    repo_id: str = typer.Option(DEFAULT_REPO_ID, help="Hugging Face repo id."),
    repo_type: str = typer.Option(DEFAULT_REPO_TYPE, help="Hugging Face repo type."),
    remote_dir: str | None = typer.Option(
        None,
        "--remote-dir",
        "-r",
        help="A file or folder/prefix to download. Defaults to HUMANO/SIMPLE resource zips.",
    ),
    local_dir: Path = typer.Option(  # noqa: B008 - Typer option declaration
        Path(DEFAULT_LOCAL_DIR),
        "--local-dir",
        "-o",
        help="Directory where files are downloaded and extracted.",
    ),
    include_scenes: bool = typer.Option(
        True,
        help="Include the default scenes from the original SIMPLE data directory.",
    ),
    all_scenes: bool = typer.Option(
        False,
        help="Download every scenes_hssd_*.zip from the dataset.",
    ),
    extract: bool = typer.Option(True, help="Extract downloaded zip files."),
    keep_zip: bool = typer.Option(False, help="Keep zip files after extraction."),
    dry_run: bool = typer.Option(False, help="Print selected files without downloading."),
    brainco: bool = typer.Option(
        False, "--brainco", help="Install only fixed official Revo2 assets for hand-type brainco."
    ),
) -> None:
    """Download HUMANO/SIMPLE data from Hugging Face.

    With no arguments this downloads the common resource zips from
    USC-PSI-Lab/SIMPLE and extracts them into the current data directory.
    """

    if brainco:
        _download_brainco(local_dir, dry_run)
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        token=_hf_token(),
    )

    if remote_dir:
        if remote_dir in repo_files:
            targets = [remote_dir]
        else:
            prefix = remote_dir.rstrip("/") + "/"
            if dry_run:
                print(f"Would download {prefix} from {repo_id}")
                return
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{remote_dir.rstrip('/')}/**",
                local_dir=str(local_dir),
                repo_type=repo_type,
                token=_hf_token(),
            )
            for filename in repo_files:
                if filename.startswith(prefix):
                    _install_downloaded_resource(local_dir, filename)
            print("Downloaded folder to:", local_path)
            return
    else:
        targets = _select_default_zips(
            repo_files,
            include_scenes=include_scenes,
            all_scenes=all_scenes,
        )

    if not targets:
        raise RuntimeError(f"No matching files found in {repo_id}")

    print(f"Downloading from {repo_type} repo {repo_id}:")
    for target in targets:
        print(f"  - {target}")

    if dry_run:
        return

    for target in targets:
        zip_path = _download_file(target, repo_id, repo_type, local_dir)
        print("Downloaded file to:", zip_path)
        if extract and zip_path.suffix == ".zip":
            _extract_zip(zip_path, local_dir, keep_zip=keep_zip)


if __name__ == "__main__":
    typer.run(download_from_huggingface)
