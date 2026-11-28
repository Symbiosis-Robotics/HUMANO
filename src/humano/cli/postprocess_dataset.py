from pathlib import Path
from typing import Annotated

import typer

from humano.datasets.lerobot import (
    PostprocessDatasetConfig,
    postprocess_dataset_with_discarded_episodes,
    postprocess_datasets,
    read_dataset_list,
)


def parse_discarded_episode_indices(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    values = tuple(item.strip() for item in value.split(","))
    if not all(values):
        raise typer.BadParameter("--discard-episodes requires comma-separated integers")
    try:
        indices = tuple(int(item) for item in values)
    except ValueError as error:
        raise typer.BadParameter("--discard-episodes requires integers") from error
    if any(index < 0 for index in indices):
        raise typer.BadParameter("--discard-episodes requires non-negative integers")
    return indices


def main(
    dataset_path: Annotated[list[Path] | None, typer.Option()] = None,
    output_path: Annotated[Path | None, typer.Option()] = None,
    dataset_list: Annotated[Path | None, typer.Option()] = None,
    discard_episodes: Annotated[str | None, typer.Option("--discard-episodes")] = None,
    remove_discarded: Annotated[
        bool, typer.Option("--remove-discarded/--no-remove-discarded")
    ] = True,
    remove_stale_smpl: Annotated[
        bool, typer.Option("--remove-stale-smpl/--no-remove-stale-smpl")
    ] = True,
) -> None:
    listed_paths = () if dataset_list is None else read_dataset_list(dataset_list)
    dataset_paths = tuple(dataset_path or ()) + listed_paths
    if output_path is None:
        raise typer.BadParameter("--output-path is required")
    discarded_episode_indices = parse_discarded_episode_indices(discard_episodes)
    config = PostprocessDatasetConfig(
        dataset_paths=dataset_paths,
        output_path=output_path,
        remove_discarded=remove_discarded,
        remove_stale_smpl=remove_stale_smpl,
    )
    summary = (
        postprocess_dataset_with_discarded_episodes(config, discarded_episode_indices)
        if discarded_episode_indices
        else postprocess_datasets(config)
    )
    typer.echo(
        f"Wrote {summary.kept_episodes} episodes to {summary.output_path} "
        f"(discarded={summary.discarded_episodes}, removed_frames={summary.removed_frames})"
    )


def typer_main() -> None:
    typer.run(main)


if __name__ == "__main__":
    typer_main()
