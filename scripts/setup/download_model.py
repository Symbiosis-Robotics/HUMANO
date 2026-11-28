#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from requests.exceptions import ConnectionError, ReadTimeout, SSLError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GEAR-SONIC deployment files from Hugging Face."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="nvidia/GEAR-SONIC",
        help="Hugging Face repo id.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default="third_party/gear_sonic_deploy",
        help="Local root directory for deployment files.",
    )

    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://huggingface.co",
        help='HF endpoint, e.g. "https://huggingface.co" or "https://hf-mirror.com".',
    )

    parser.add_argument(
        "--proxy",
        type=str,
        default="",
        help="HTTP/HTTPS proxy. Empty string disables proxy.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HF repo revision / branch / commit.",
    )

    parser.add_argument(
        "--num-retries",
        type=int,
        default=30,
        help="Number of retry attempts.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what will be downloaded.",
    )

    return parser.parse_args()


def setup_env(args):
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1600"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "160"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    else:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)


def download_one(repo_id: str, remote_file: str, dst: Path, revision: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=remote_file,
            repo_type="model",
            revision=revision,
            token=os.environ.get("HF_TOKEN", None),
        )
    )

    shutil.copy2(cached, dst)
    print(f"[OK] {remote_file} -> {dst}")


def main():
    args = parse_args()
    setup_env(args)

    root = Path(args.root).expanduser().resolve()

    files = {
        "model_encoder.onnx": "policy/release/model_encoder.onnx",
        "model_decoder.onnx": "policy/release/model_decoder.onnx",
        "observation_config.yaml": "policy/release/observation_config.yaml",
        "planner_sonic.onnx": "planner/target_vel/V2/planner_sonic.onnx",
    }

    print("Download config:")
    print(f"  repo_id     = {args.repo_id}")
    print(f"  root        = {root}")
    print(f"  hf_endpoint = {args.hf_endpoint}")
    print(f"  proxy       = {args.proxy or '<disabled>'}")
    print(f"  revision    = {args.revision}")
    print("  files:")
    for remote, local in files.items():
        print(f"    {remote} -> {root / local}")

    if args.dry_run:
        print("[DRY RUN] Exit without downloading.")
        return

    for remote_file, local_path in files.items():
        dst = root / local_path

        for attempt in range(args.num_retries):
            try:
                download_one(
                    repo_id=args.repo_id,
                    remote_file=remote_file,
                    dst=dst,
                    revision=args.revision,
                )
                break

            except (
                HfHubHTTPError,
                LocalEntryNotFoundError,
                SSLError,
                ConnectionError,
                ReadTimeout,
            ) as e:
                msg = str(e)

                if "401" in msg or "Unauthorized" in msg:
                    raise RuntimeError(
                        "401 Unauthorized: 请设置有效 HF_TOKEN，或确认该 repo/文件可公开访问。"
                    ) from e

                if "404" in msg or "Entry Not Found" in msg or "not found" in msg.lower():
                    raise RuntimeError(f"文件不存在：{remote_file} in {args.repo_id}") from e

                if "429" in msg or "Too Many Requests" in msg:
                    wait_time = 360
                    print(
                        f"[RATE LIMIT] {remote_file}: attempt {attempt + 1}/{args.num_retries} failed."
                    )
                else:
                    wait_time = min(300, 30 * (attempt + 1))
                    print(f"[WARN] {remote_file}: attempt {attempt + 1}/{args.num_retries} failed.")

                print(repr(e))
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        else:
            raise RuntimeError(
                f"Failed to download {remote_file} after {args.num_retries} attempts."
            )

    print("[OK] All GEAR-SONIC deployment files downloaded successfully.")


if __name__ == "__main__":
    main()
