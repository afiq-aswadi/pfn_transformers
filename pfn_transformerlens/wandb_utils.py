"""Utilities for loading models from Weights & Biases."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal
import os

import torch

from pfn_transformerlens.model.PFN import BasePFN
from pfn_transformerlens.checkpointing import load_checkpoint, CheckpointMetadata


@dataclass
class ModelInfo:
    """Model metadata from wandb."""

    run_id: str
    run_name: str
    run_url: str
    checkpoint_step: int
    created_at: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any] | None


@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint artifact."""

    step: int
    artifact_name: str
    created_at: str


@dataclass
class RunNameScheme:
    """Defines which config fields to include in run names and use for filtering.

    Example:
        >>> scheme = RunNameScheme.from_templates(
        ...     model={'n_layers': None, 'd_model': None},
        ...     data={'num_tasks': None}
        ... )
        >>> scheme.model_fields
        ('n_layers', 'd_model')
    """

    model_fields: tuple[str, ...] = ()
    training_fields: tuple[str, ...] = ()
    data_fields: tuple[str, ...] = ()

    @classmethod
    def from_templates(
        cls,
        model: Any | None = None,
        training: Any | None = None,
        data: Any | None = None,
    ) -> "RunNameScheme":
        """Create scheme from template configs or dicts.

        Pass dicts or dataclass instances with only the fields you care about.
        Values don't matter, only keys/field names are extracted.

        Args:
            model: Model config template (dict or dataclass)
            training: Training config template (dict or dataclass)
            data: Data config template (dict or dataclass)

        Returns:
            RunNameScheme with extracted field names

        Example:
            >>> scheme = RunNameScheme.from_templates(
            ...     model={'n_layers': None, 'd_model': None, 'mask_type': None},
            ...     data={'num_tasks': None}
            ... )
        """
        return cls(
            model_fields=tuple(_extract_fields_from_template(model)),
            training_fields=tuple(_extract_fields_from_template(training)),
            data_fields=tuple(_extract_fields_from_template(data)),
        )


class WandbAdapter:
    """Thin adapter around wandb API for testability."""

    def __init__(self):
        try:
            import wandb

            self.wandb = wandb
            self.api = wandb.Api()
        except ImportError as e:
            raise ImportError(
                "wandb required for model loading. Install with: uv sync --extra wandb"
            ) from e

    def get_run(self, path: str):
        """Fetch a single run by fully-qualified path."""
        return self.api.run(path)

    def get_runs(self, path: str, filters: dict):
        """Query wandb runs with filters."""
        return self.api.runs(path, filters=filters)

    def get_artifact(self, name: str):
        """Get artifact by name."""
        return self.api.artifact(name)


def load_from_pretrained(
    run_identifier: str,
    checkpoint_step: int | None = None,
    project: str | None = None,
    entity: str | None = None,
    device: str = "auto",
    load_optimizer: bool = False,
    adapter: WandbAdapter | None = None,
) -> tuple[BasePFN, dict | None, CheckpointMetadata]:
    """
    Load model from wandb artifact.

    Uses wandb's built-in artifact caching (~/.cache/wandb/).
    Does NOT delete cached files after loading.

    Args:
        run_identifier: wandb run ID (recommended) or run name
        checkpoint_step: Specific step number, None for latest
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)
        device: Device to load on. "auto" = cuda > mps > cpu
        load_optimizer: If True, return optimizer state dict
        adapter: WandbAdapter for testing (internal use)

    Returns:
        Tuple of (model, optimizer_state | None, metadata)
        - model: Loaded BasePFN in eval mode on specified device
        - optimizer_state: State dict if load_optimizer=True, else None
        - metadata: CheckpointMetadata with wandb/git info

    Raises:
        ImportError: If wandb not installed
        ValueError: If run/artifact not found

    Example:
        >>> model, _, metadata = load_from_pretrained(
        ...     "abc123",
        ...     checkpoint_step=5000,
        ...     project="pfn-experiments",
        ...     entity="myteam"
        ... )
        >>> print(f"Loaded from {metadata.wandb_run_url}")
    """
    adapter = adapter or WandbAdapter()
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")

    run = _resolve_run(adapter, run_identifier, project, entity)

    # Find artifact
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]

    if checkpoint_step is not None:
        # Find specific step using run.id-based naming
        target_name = f"checkpoint-{run.id}-step-{checkpoint_step}"
        artifact = next((a for a in artifacts if target_name in a.name), None)
        if not artifact:
            available_steps = [_extract_step(a.name) for a in artifacts]
            raise ValueError(
                f"No checkpoint at step {checkpoint_step}. "
                f"Available steps: {available_steps}"
            )
    else:
        # Latest by created_at
        if not artifacts:
            raise ValueError(f"No model artifacts for run {run.name}")
        artifact = max(artifacts, key=lambda a: a.created_at)

    # Download artifact (wandb caches, we don't delete)
    artifact_dir = artifact.download()
    checkpoint_files = list(Path(artifact_dir).glob("*.pt"))

    if not checkpoint_files:
        raise ValueError(f"No .pt checkpoint file found in artifact {artifact.name}")

    checkpoint_path = checkpoint_files[0]

    # Load using checkpointing module
    model, optimizer_state, metadata = load_checkpoint(
        checkpoint_path, device=device, load_optimizer=load_optimizer
    )

    return model, optimizer_state, metadata


def list_available_models(
    project: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
    adapter: WandbAdapter | None = None,
) -> list[ModelInfo]:
    """
    List available models in wandb project.

    Args:
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)
        tags: Filter runs by tags
        adapter: WandbAdapter for testing (internal use)

    Returns:
        List of ModelInfo with run metadata

    Example:
        >>> models = list_available_models(
        ...     project="pfn-experiments",
        ...     entity="myteam",
        ...     tags=["production"]
        ... )
        >>> for model in models:
        ...     print(f"{model.run_name}: {model.checkpoint_step} steps")
    """
    adapter = adapter or WandbAdapter()
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")

    if not project or not entity:
        raise ValueError(
            "Must provide project/entity or set WANDB_PROJECT/WANDB_ENTITY env vars"
        )

    # Build filters
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}

    runs = adapter.get_runs(f"{entity}/{project}", filters=filters)

    model_entries: list[tuple[float | str, ModelInfo]] = []
    for run in runs:
        # Get artifacts for this run
        artifacts = [a for a in run.logged_artifacts() if a.type == "model"]

        if not artifacts:
            continue

        # Get latest artifact
        latest = max(artifacts, key=lambda a: a.created_at)
        step = _extract_step(latest.name)

        # Read configs from artifact metadata (not run.config)
        # This is more reliable as it's stored directly with the checkpoint
        artifact_meta = latest.metadata or {}
        model_config = _to_mapping(artifact_meta.get("model_config", {}))
        training_config = _to_mapping(artifact_meta.get("training_config", {}))
        data_config = _to_mapping(artifact_meta.get("data_config")) or None

        created_at_raw = latest.created_at
        created_at_str = _format_timestamp(created_at_raw)
        sort_key_method = getattr(created_at_raw, "timestamp", None)
        sort_key: float | str
        if callable(sort_key_method):
            sort_key = float(sort_key_method())
        else:
            sort_key = created_at_str

        model_entries.append(
            (
                sort_key,
                ModelInfo(
                    run_id=run.id,
                    run_name=run.name,
                    run_url=run.url,
                    checkpoint_step=step,
                    created_at=created_at_str,
                    model_config=model_config,
                    training_config=training_config,
                    data_config=data_config,
                ),
            )
        )

    model_entries.sort(key=lambda item: item[0], reverse=True)
    return [entry[1] for entry in model_entries]


def get_checkpoint_metadata(
    run_identifier: str,
    checkpoint_step: int | None = None,
    project: str | None = None,
    entity: str | None = None,
    adapter: WandbAdapter | None = None,
) -> CheckpointMetadata:
    """
    Get checkpoint metadata without downloading artifact.

    Useful for inspecting model info before loading.

    Args:
        run_identifier: wandb run ID or run name
        checkpoint_step: Specific step number, None for latest
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)
        adapter: WandbAdapter for testing (internal use)

    Returns:
        CheckpointMetadata from the checkpoint

    Raises:
        ValueError: If run/artifact not found

    Example:
        >>> metadata = get_checkpoint_metadata("abc123", checkpoint_step=5000)
        >>> print(f"Trained on: {metadata.timestamp}")
        >>> print(f"Git hash: {metadata.git_hash}")
    """
    adapter = adapter or WandbAdapter()
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")

    if not project or not entity:
        raise ValueError(
            "Must provide project/entity or set WANDB_PROJECT/WANDB_ENTITY env vars"
        )

    run = _resolve_run(adapter, run_identifier, project, entity)

    # Find artifact
    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]

    if checkpoint_step is not None:
        target_name = f"checkpoint-{run.id}-step-{checkpoint_step}"
        artifact = next((a for a in artifacts if target_name in a.name), None)
        if not artifact:
            raise ValueError(f"No checkpoint at step {checkpoint_step}")
    else:
        if not artifacts:
            raise ValueError(f"No model artifacts for run {run.name}")
        artifact = max(artifacts, key=lambda a: a.created_at)

    # Extract metadata from artifact metadata (not the checkpoint file)
    # This avoids downloading the full checkpoint
    meta = artifact.metadata or {}
    timestamp = meta.get("timestamp")
    git_hash = meta.get("git_hash")
    wandb_run_id = meta.get("wandb_run_id", run.id)
    wandb_run_name = meta.get("wandb_run_name", run.name)
    wandb_run_url = meta.get("wandb_run_url", run.url)

    if timestamp is not None or git_hash is not None:
        return CheckpointMetadata(
            timestamp=timestamp or "",
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
            wandb_run_url=wandb_run_url,
            git_hash=git_hash,
        )

    # fall back to reading checkpoint metadata if artifact metadata missing entries
    artifact_dir = artifact.download()
    checkpoint_files = list(Path(artifact_dir).glob("*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No .pt checkpoint file found in artifact {artifact.name}")

    ckpt = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
    metadata_dict = ckpt.get("metadata", {})
    return CheckpointMetadata(
        timestamp=str(metadata_dict.get("timestamp", "")),
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
        wandb_run_url=wandb_run_url,
        git_hash=metadata_dict.get("git_hash"),
    )


def list_checkpoints(
    run_identifier: str,
    project: str | None = None,
    entity: str | None = None,
    adapter: WandbAdapter | None = None,
) -> list[CheckpointInfo]:
    """
    List all checkpoint artifacts for a given run.

    Args:
        run_identifier: wandb run ID, name, or fully-qualified path
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)
        adapter: WandbAdapter for testing (internal use)

    Returns:
        List of CheckpointInfo sorted by step ascending.
    """
    adapter = adapter or WandbAdapter()
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")

    run = _resolve_run(adapter, run_identifier, project, entity)
    artifacts = [
        artifact for artifact in run.logged_artifacts() if artifact.type == "model"
    ]
    checkpoint_infos = [
        CheckpointInfo(
            step=_extract_step(artifact.name),
            artifact_name=artifact.name,
            created_at=str(artifact.created_at),
        )
        for artifact in artifacts
    ]
    checkpoint_infos.sort(key=lambda info: info.step)
    return checkpoint_infos


def load_by_config(
    scheme: RunNameScheme,
    checkpoint_step: int | Literal["latest", "earliest"] | None = "latest",
    device: str = "auto",
    load_optimizer: bool = False,
    project: str | None = None,
    entity: str | None = None,
    adapter: WandbAdapter | None = None,
    **config_filters: Any,
) -> tuple[BasePFN, dict | None, CheckpointMetadata]:
    """
    Load model by matching config values defined in scheme.

    Only filters on fields specified in the scheme. Other config fields are ignored.

    Args:
        scheme: RunNameScheme defining which fields to filter on
        checkpoint_step: Step number, "latest" (default), or "earliest"
        device: Device to load on. "auto" = cuda > mps > cpu
        load_optimizer: If True, return optimizer state dict
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)
        adapter: WandbAdapter for testing (internal use)
        **config_filters: Config field values to match (e.g., n_layers=4, mask_type="causal")

    Returns:
        Tuple of (model, optimizer_state | None, metadata)

    Raises:
        ValueError: If no matches or multiple matches found

    Example:
        >>> scheme = RunNameScheme.from_templates(
        ...     model={'n_layers': None, 'mask_type': None},
        ...     data={'num_tasks': None}
        ... )
        >>> model, _, meta = load_by_config(
        ...     scheme,
        ...     n_layers=4,
        ...     mask_type="causal",
        ...     num_tasks=10,
        ...     checkpoint_step="latest"
        ... )
    """
    adapter = adapter or WandbAdapter()
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")

    if not project or not entity:
        raise ValueError(
            "Must provide project/entity or set WANDB_PROJECT/WANDB_ENTITY env vars"
        )

    # Get all models and filter by config
    all_models = list_available_models(project=project, entity=entity, adapter=adapter)
    matching_models = [
        m for m in all_models if _matches_scheme_filters(m, scheme, config_filters)
    ]

    if not matching_models:
        raise ValueError(
            f"No models found matching config filters: {config_filters}\n"
            f"Scheme fields: model={scheme.model_fields}, "
            f"training={scheme.training_fields}, data={scheme.data_fields}"
        )

    # Apply checkpoint_step filtering/tiebreaking
    if isinstance(checkpoint_step, int):
        # Filter to runs with this exact step
        filtered = []
        for m in matching_models:
            checkpoints = list_checkpoints(m.run_id, project, entity, adapter)
            if any(ckpt.step == checkpoint_step for ckpt in checkpoints):
                filtered.append(m)
        matching_models = filtered

        if not matching_models:
            raise ValueError(
                f"No models with checkpoint at step {checkpoint_step} "
                f"matching config filters: {config_filters}"
            )

    # Pick based on tiebreak strategy
    if len(matching_models) == 1:
        selected = matching_models[0]
    elif checkpoint_step == "latest":
        # Pick most recent by created_at
        selected = matching_models[0]  # list_available_models already sorts by latest
    elif checkpoint_step == "earliest":
        # Pick oldest
        selected = matching_models[-1]
    else:
        # Multiple matches, show options
        raise ValueError(
            f"Multiple models ({len(matching_models)}) match config filters.\n"
            f"Matching models:\n{_format_model_table(matching_models)}\n\n"
            f"Specify a more specific config filter or use checkpoint_step with an int."
        )

    # Load the selected model
    step_to_load = checkpoint_step if isinstance(checkpoint_step, int) else None
    return load_from_pretrained(
        run_identifier=selected.run_id,
        checkpoint_step=step_to_load,
        project=project,
        entity=entity,
        device=device,
        load_optimizer=load_optimizer,
        adapter=adapter,
    )


_DEFAULT_RUN_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "model": ("d_model", "n_layers", "n_heads"),
    "training": ("learning_rate", "num_steps", "batch_size"),
    "data": ("num_tasks", "task_diversity"),
}


def create_run_name(
    *,
    base: str,
    model_config: Any | None = None,
    training_config: Any | None = None,
    data_config: Any | None = None,
    scheme: RunNameScheme | None = None,
    include_fields: dict[str, Sequence[str]] | None = None,
    extra: Mapping[str, Any] | None = None,
    max_length: int = 128,
) -> str:
    """
    Construct a wandb run name from selected configuration fields.

    Args:
        base: Prefix for the run name (e.g., task identifier).
        model_config: Model configuration dataclass or mapping.
        training_config: Training configuration dataclass or mapping.
        data_config: Data/generator configuration dataclass or mapping.
        scheme: RunNameScheme defining which fields to include (overrides include_fields).
        include_fields: Optional overrides for which fields to include per section.
        extra: Additional key/value pairs to append to the name.
        max_length: Maximum length of the resulting name.

    Returns:
        Sanitised run name string containing the selected fields.

    Example:
        >>> scheme = RunNameScheme.from_templates(
        ...     model={'n_layers': None, 'd_model': None},
        ...     data={'num_tasks': None}
        ... )
        >>> create_run_name(base="pfn", model_config=cfg, scheme=scheme)
    """
    if not base:
        raise ValueError("base must be a non-empty string")

    # Convert scheme to include_fields format
    if scheme is not None:
        field_map = {
            "model": scheme.model_fields,
            "training": scheme.training_fields,
            "data": scheme.data_fields,
        }
    else:
        field_map = include_fields or _DEFAULT_RUN_NAME_FIELDS
    sections: dict[str, tuple[Any | None, Sequence[str]]] = {
        "model": (model_config, field_map.get("model", ())),
        "training": (training_config, field_map.get("training", ())),
        "data": (data_config, field_map.get("data", ())),
    }

    components: list[str] = [_slugify(base)]
    for section, (src, fields) in sections.items():
        if src is None or not fields:
            continue
        values = _extract_fields(src, fields)
        if not values:
            continue
        tokens: list[str] = [section]
        for field in fields:
            if field not in values:
                continue
            value = values[field]
            token = f"{field}{_normalise_value(value)}"
            tokens.append(token)
        components.append(_slugify("-".join(tokens)))

    if extra:
        for key, value in extra.items():
            token = f"{key}{_normalise_value(value)}"
            components.append(_slugify(token))

    run_name = "-".join(filter(None, components))
    if len(run_name) > max_length:
        run_name = run_name[:max_length].rstrip("-")
    return run_name


def _extract_step(artifact_name: str) -> int:
    """Extract step number from artifact name.

    Args:
        artifact_name: Artifact name like "checkpoint-{run_id}-step-{step}"

    Returns:
        Step number as int
    """
    # artifact_name format: "checkpoint-{run_id}-step-{step}:v{version}"
    # Split by "-step-" and take the last part, then remove version suffix
    try:
        step_part = artifact_name.split("-step-")[-1]
        # Remove version suffix (e.g., ":v0")
        step_str = step_part.split(":")[0]
        return int(step_str)
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Could not extract step from artifact name: {artifact_name}"
        ) from e


def _extract_fields(source: Any, fields: Sequence[str]) -> dict[str, Any]:
    data = _to_mapping(source)
    return {field: data[field] for field in fields if field in data}


def _extract_fields_from_template(template: Any | None) -> list[str]:
    """Extract field names from a template dict or dataclass instance.

    Args:
        template: Dict or dataclass instance (values don't matter)

    Returns:
        List of field names
    """
    if template is None:
        return []

    if isinstance(template, Mapping):
        return list(template.keys())

    if is_dataclass(template):
        from dataclasses import fields

        return [f.name for f in fields(template)]

    if hasattr(template, "__dict__"):
        return [k for k in vars(template).keys() if not k.startswith("_")]

    return []


def _to_mapping(source: Any | None) -> dict[str, Any]:
    if source is None:
        return {}
    if is_dataclass(source):
        return dict(asdict(source))
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "__dict__"):
        return {
            key: value for key, value in vars(source).items() if not key.startswith("_")
        }
    return {}


def _normalise_value(value: Any) -> str:
    if isinstance(value, float):
        formatted = f"{value:.4g}"
        return formatted.rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return str(value)


def _slugify(token: str) -> str:
    lowered = token.lower()
    cleaned: list[str] = []
    last_was_dash = False
    for char in lowered:
        if char.isalnum() or char in {"_", "."}:
            cleaned.append(char)
            last_was_dash = False
        else:
            if not last_was_dash:
                cleaned.append("-")
                last_was_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "run"


def _resolve_run(
    adapter: WandbAdapter,
    run_identifier: str,
    project: str | None,
    entity: str | None,
):
    """
    Resolve a wandb run either by full path, run ID, or display name.

    Args:
        adapter: WandbAdapter instance
        run_identifier: run ID, display name, or fully-qualified path
        project: wandb project
        entity: wandb entity

    Returns:
        wandb.apis.public.Run instance
    """
    if "/" in run_identifier:
        # assume fully-qualified path: entity/project/run_id_or_name
        try:
            return adapter.get_run(run_identifier)
        except adapter.wandb.errors.CommError as exc:
            raise ValueError(f"No run found at path {run_identifier}") from exc

    if not project or not entity:
        raise ValueError(
            "Must provide project/entity or set WANDB_PROJECT/WANDB_ENTITY env vars"
        )

    # first try treating identifier as run id
    try:
        return adapter.get_run(f"{entity}/{project}/{run_identifier}")
    except adapter.wandb.errors.CommError:
        pass

    # fallback: search by display name
    runs = adapter.get_runs(
        f"{entity}/{project}",
        filters={"displayName": run_identifier},
    )

    if not runs:
        raise ValueError(f"No run found: {run_identifier}")
    if len(runs) > 1:
        raise ValueError(
            f"Multiple runs found with name '{run_identifier}'. "
            "Provide the run ID or full path instead."
        )

    return runs[0]


def _matches_scheme_filters(
    model: ModelInfo, scheme: RunNameScheme, filters: dict[str, Any]
) -> bool:
    """Check if a model matches all config filters defined in scheme.

    Args:
        model: ModelInfo to check
        scheme: RunNameScheme defining which fields to check
        filters: Dict of field_name -> expected_value

    Returns:
        True if all filters match, False otherwise
    """
    # Build flat config from model
    flat_config: dict[str, Any] = {}

    # Add model fields
    for field in scheme.model_fields:
        if field in model.model_config:
            flat_config[field] = model.model_config[field]

    # Add training fields
    for field in scheme.training_fields:
        if field in model.training_config:
            flat_config[field] = model.training_config[field]

    # Add data fields
    if model.data_config:
        for field in scheme.data_fields:
            if field in model.data_config:
                flat_config[field] = model.data_config[field]

    # Check all filters match
    for field, expected_value in filters.items():
        if field not in flat_config:
            return False
        if flat_config[field] != expected_value:
            return False

    return True


def _format_model_table(models: list[ModelInfo]) -> str:
    """Format list of models into a readable table.

    Args:
        models: List of ModelInfo to format

    Returns:
        Formatted string table
    """
    lines = []
    lines.append(
        f"{'Run ID':<20} {'Run Name':<40} {'Step':<8} {'Created':<20} {'URL':<60}"
    )
    lines.append("-" * 150)

    for m in models:
        run_id = m.run_id[:17] + "..." if len(m.run_id) > 20 else m.run_id
        run_name = m.run_name[:37] + "..." if len(m.run_name) > 40 else m.run_name
        created = m.created_at[:20]
        url = m.run_url[:57] + "..." if len(m.run_url) > 60 else m.run_url

        lines.append(
            f"{run_id:<20} {run_name:<40} {m.checkpoint_step:<8} {created:<20} {url:<60}"
        )

    return "\n".join(lines)


__all__ = [
    "WandbAdapter",
    "ModelInfo",
    "CheckpointInfo",
    "RunNameScheme",
    "load_from_pretrained",
    "load_by_config",
    "list_available_models",
    "get_checkpoint_metadata",
    "list_checkpoints",
    "create_run_name",
]
