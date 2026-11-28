# Task instructions

Each task directory has its own `meta/tasks.jsonl` mapping from `task_index` to instruction text. Resolve indices within the same task directory and split.

## Training

LeRobot resolves each Parquet row's `task_index` through `meta/tasks.jsonl` and returns the text as `sample["task"]`. Pass this text to the policy's language encoder. For Goal and Object tasks it identifies the selected target. For Condition tasks it preserves the full conditional rule. For Long tasks it preserves the required action order.

## Evaluation

Set `HUMANO_EVAL_INSTRUCTION_SOURCE=data` when launching evaluation. For ID and non-language OOD episodes, the evaluator reads the episode instruction from `tasks[0]` in `meta/episodes.jsonl`. For Language OOD it reads the saved paraphrase from `environment_config.dr_state_dict.language`. A generic task description or a fixed training prompt loses this episode-specific information.

When reading evaluation metadata yourself, parse the JSON string in `environment_config` before accessing its nested fields. For Language OOD, `environment_config.ood.params.source_instruction` records the original wording when present.
