"""Загрузка и валидация конфигурации проекта."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


@dataclass
class PathsConfig:
    data_dir: Path
    artifacts_dir: Path


@dataclass
class ModelSpec:
    name: str
    architecture: str
    description: str
    encoder_name: Optional[str]
    checkpoint: Path


@dataclass
class ModelConfig:
    active: str
    num_classes: int
    registry: Dict[str, ModelSpec]


@dataclass
class InferenceConfig:
    image_size: Tuple[int, int]
    threshold: float
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]


@dataclass
class TrainingConfig:
    image_size: Tuple[int, int]
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    num_workers: int
    scheduler_factor: float
    scheduler_patience: int
    seed: int


@dataclass
class ServiceConfig:
    host: str
    port: int
    log_level: str


@dataclass
class Config:
    paths: PathsConfig
    model: ModelConfig
    inference: InferenceConfig
    training: TrainingConfig
    service: ServiceConfig
    project_root: Path = field(default_factory=lambda: Path.cwd())

    def resolve_path(self, raw: str | Path) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def get_model_spec(self, name: str | None = None) -> ModelSpec:
        key = name or self.model.active
        if key not in self.model.registry:
            available = ", ".join(sorted(self.model.registry))
            raise ValueError(f"Unknown model '{key}'. Available: {available}")
        return self.model.registry[key]

    def active_checkpoint(self) -> Path:
        return self.resolve_path(self.get_model_spec().checkpoint)


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


def _project_root_from(config_path: Path) -> Path:
    """Корнем проекта считаем папку, содержащую configs/ и src/."""
    config_path = config_path.resolve()
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return config_path.parent.parent


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.environ.get("POLYP_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = _project_root_from(config_path)

    paths = PathsConfig(
        data_dir=Path(raw["paths"]["data_dir"]),
        artifacts_dir=Path(raw["paths"]["artifacts_dir"]),
    )

    registry: Dict[str, ModelSpec] = {}
    for name, spec in raw["models"].items():
        registry[name] = ModelSpec(
            name=name,
            architecture=spec["architecture"],
            description=spec.get("description", ""),
            encoder_name=spec.get("encoder_name"),
            checkpoint=Path(spec["checkpoint"]),
        )

    model = ModelConfig(
        active=raw["model"]["active"],
        num_classes=int(raw["model"]["num_classes"]),
        registry=registry,
    )

    inf = raw["inference"]
    inference = InferenceConfig(
        image_size=tuple(inf["image_size"]),
        threshold=float(inf["threshold"]),
        normalize_mean=tuple(inf["normalize_mean"]),
        normalize_std=tuple(inf["normalize_std"]),
    )

    tr = raw["training"]
    training = TrainingConfig(
        image_size=tuple(tr["image_size"]),
        batch_size=int(tr["batch_size"]),
        learning_rate=float(tr["learning_rate"]),
        weight_decay=float(tr["weight_decay"]),
        epochs=int(tr["epochs"]),
        num_workers=int(tr["num_workers"]),
        scheduler_factor=float(tr["scheduler_factor"]),
        scheduler_patience=int(tr["scheduler_patience"]),
        seed=int(tr.get("seed", 42)),
    )
    service = ServiceConfig(**raw["service"])

    cfg = Config(
        paths=paths,
        model=model,
        inference=inference,
        training=training,
        service=service,
        project_root=project_root,
    )

    env_active = os.environ.get("POLYP_MODEL_ACTIVE")
    if env_active:
        cfg.model.active = env_active

    return cfg
