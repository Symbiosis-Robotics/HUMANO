"""Resolve frozen OOD text before legacy training/dataset text selection."""


def resolve_eval_instruction(task, environment_config, episode_metadata, source=""):
    variant = environment_config.get("ood", {})
    if variant.get("dimension") == "language":
        instruction = environment_config.get("dr_state_dict", {}).get("language")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Language OOD requires a saved non-empty instruction")
        return instruction
    if source in {"description", "native_description"}:
        return task.description
    if source in {"data", "episode_tasks"}:
        texts = episode_metadata.get("tasks")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Instruction source=data requires non-empty episode tasks")
        return str(texts[0])
    return task.instruction
