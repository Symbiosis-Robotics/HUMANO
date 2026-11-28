"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

import os
import shutil
import time
from datetime import datetime

import gymnasium as gym

from humano.envs.video_writer import AsyncVideoWriter, VideoWriter


class VideoRecorder(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str = "video",
        framerate: int = 10,
        # camera: List[str] = ["mujoco", "front_left", "wrist"],
        name_prefix: str | None = None,
        write_png: bool = False,
        async_write: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, video_folder=video_folder, name_prefix=name_prefix, write_png=write_png
        )
        gym.Wrapper.__init__(self, env)

        os.makedirs(video_folder, exist_ok=True)

        # self._elapsed_steps = None
        self.work_dir = video_folder

        if name_prefix is None:
            now = datetime.now()
            name_prefix = now.isoformat().replace(":", "-").replace(".", "-")
            if self.unwrapped.__module__.startswith("humano.envs"):
                self.sim_mode = self.unwrapped.sim_mode  # type: ignore
                name_prefix = f"{self.unwrapped.task.uid}_{name_prefix}"  # type: ignore

        self.name_prefix = name_prefix
        self.write_png = write_png
        self.async_write = async_write
        self.framerate = framerate
        self.video_writers = {}
        self._step_count = 0

    def reset(self, **kwargs):
        print(f"[VideoRecorder][DEBUG] Resetting environment with kwargs: {kwargs}")
        observations, info = super().reset(**kwargs)
        print(f"[VideoRecorder][DEBUG] super reset done")

        if kwargs.get("options") is not None:
            if kwargs["options"].get("task_id") is not None:
                self.name_prefix = kwargs["options"]["task_id"]  # overwrite name prefix with task_id
                video_folder = f"{self.work_dir}/{self.name_prefix}"
                if os.path.exists(video_folder):
                    shutil.rmtree(video_folder, ignore_errors=True)
                    print(f"Overwriting existing videos at {video_folder} folder")

                os.makedirs(video_folder, exist_ok=True)

        self.video_writers = {}
        _WriterCls = AsyncVideoWriter if self.async_write else VideoWriter
        for key, subspace in self.observation_space.items():
            if len(subspace.shape) == 3 and subspace.shape[-1] == 3:  # only record image observations
                self.video_writers[key] = _WriterCls(
                    f"{self.work_dir}/{self.name_prefix}/{key}.mp4",
                    self.framerate,
                    subspace.shape[:2][::-1],
                    write_png=self.write_png,
                )
                self.video_writers[key].write(observations[key])

        self._is_released = False
        return observations, info

    def render(self):
        pass

    def step(self, action):
        t_start = time.monotonic()
        observation, reward, terminated, truncated, info = self.env.step(action)
        t_inner = time.monotonic()
        for key, video_writer in self.video_writers.items():
            video_writer.write(observation[key])
        t_write = time.monotonic()
        self._step_count += 1
        if self._step_count <= 3 or self._step_count % 200 == 0 or (t_write - t_start) > 0.005:
            inner_ms = (t_inner - t_start) * 1000.0
            write_ms = (t_write - t_inner) * 1000.0
            mode = "async_q" if self.async_write else "sync"
            print(
                f"[VideoRecorder] step={self._step_count} "
                f"inner_env={inner_ms:.1f}ms video_write({mode})={write_ms:.1f}ms",
                flush=True,
            )

        return observation, reward, terminated, truncated, info

    def release(self):
        if not self._is_released:
            for video_writer in self.video_writers.values():
                video_writer.release(self.unwrapped._success)  # type: ignore
            self._is_released = True

    def close(self):
        """Closes the wrapper then the video recorder."""
        self.release()
        super().close()
