"""Local model management utilities for the Little Town prototype.

The goal of this helper module is to offer a batteries-included experience for
non-technical players: the launcher can automatically provision trusted local
language models that run through ``llama.cpp`` bindings without requiring the
user to touch the command line.  The default registry focuses on permissively
licensed models that are widely regarded as safe community choices (avoiding
sources like DeepSeek, per the project brief).

The download logic is intentionally defensive: it verifies hashes when
available, stores metadata about each managed artefact, and gracefully degrades
whenever the machine is offline or the download is skipped through an
environment override.  This keeps the launcher snappy for quick smoke tests
while still supporting the desired one-click install path for hobbyists.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # ``huggingface_hub`` is relatively small and pure Python.
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - handled by the launcher
    hf_hub_download = None  # type: ignore[assignment]


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a downloadable local model artefact."""

    name: str
    repo_id: str
    filename: str
    license: str
    description: str
    default: bool = True
    sha256: Optional[str] = None
    provider: str = "community"

    def to_dict(self) -> Dict[str, str]:
        data = dataclasses.asdict(self)
        # Remove ``None`` values for cleaner metadata files.
        return {k: v for k, v in data.items() if v is not None}


@dataclasses.dataclass
class ModelStatus:
    """Represents the availability of a model on disk."""

    spec: ModelSpec
    path: Optional[Path]
    state: str
    message: str = ""

    def as_summary(self) -> Dict[str, str]:
        return {
            "name": self.spec.name,
            "state": self.state,
            "path": str(self.path) if self.path else "",
            "message": self.message,
        }


TRUSTED_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="TinyLlama 1.1B Chat Q4_K_M",
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        license="Apache-2.0",
        description=(
            "A compact quantised TinyLlama build that runs comfortably on mid-range"
            " CPUs via llama.cpp bindings. Suitable for dialogue, journaling, and"
            " lightweight planning support inside the simulation."
        ),
        # sha256 from Hugging Face file metadata. Included so we can verify the
        # download when available.
        sha256="8cb96d08d7f3f7319e4470612b4c39719695f637ed2db3346ac8891a5fa0aaf6",
        provider="TinyLlama",
    ),
    ModelSpec(
        name="Phi-2 Q4_K_M",
        repo_id="TheBloke/phi-2-GGUF",
        filename="phi-2.Q4_K_M.gguf",
        license="MIT",
        description=(
            "Phi-2 distilled for llama.cpp. Offers strong reasoning performance"
            " while staying within the memory budget of most hobbyist machines."
        ),
        sha256="4103a7db77ae16d3a9c168df2fb23458b699d1f6d086dc3848d1b5da760c01dd",
        provider="Microsoft",
    ),
]


COMMUNITY_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="Mistral 7B Instruct Q4_K_M",
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        license="Apache-2.0",
        description=(
            "Mistral's lightweight instruct-tuned release in llama.cpp format."
            " Provides higher quality dialogue and planning without needing a"
            " dedicated GPU."
        ),
        default=False,
        provider="Mistral",
    ),
    ModelSpec(
        name="Xwin LM 7B V0.2 Q4_K_M",
        repo_id="TheBloke/Xwin-LM-7B-V0.2-GGUF",
        filename="xwin-lm-7b-v0.2.Q4_K_M.gguf",
        license="Apache-2.0",
        description=(
            "A community maintained derivative inspired by xAI research notes."
            " Good generalist assistant with balanced creativity."
        ),
        default=False,
        provider="xAI community",
    ),
    ModelSpec(
        name="OpenHermes 2.5 Mistral Q4_K_M",
        repo_id="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        filename="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        license="Apache-2.0",
        description=(
            "An open instruction model inspired by OpenAI conversational flows."
            " Provides more roleplay-oriented responses for town memories."
        ),
        default=False,
        provider="OpenHermes",
    ),
]


class LocalModelManager:
    """Handles provisioning, verification, and discovery of local LLM assets."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path.cwd() / "Assets" / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.root / "model_registry.json"
        self._catalog_path = self.root / "custom_models.json"
        self._lock = threading.Lock()
        self._custom_specs = self._load_catalog()

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        if not self._metadata_path.exists():
            return {}
        try:
            return json.loads(self._metadata_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_metadata(self, entries: Iterable[ModelStatus]) -> None:
        data = {status.spec.filename: status.as_summary() for status in entries}
        self._metadata_path.write_text(json.dumps(data, indent=2))

    def _load_catalog(self) -> Dict[str, ModelSpec]:
        if not self._catalog_path.exists():
            return {}
        try:
            raw = json.loads(self._catalog_path.read_text())
        except json.JSONDecodeError:
            return {}
        specs: Dict[str, ModelSpec] = {}
        for entry in raw:
            try:
                specs[entry["filename"]] = ModelSpec(
                    name=entry["name"],
                    repo_id=entry["repo_id"],
                    filename=entry["filename"],
                    license=entry.get("license", ""),
                    description=entry.get("description", "Custom community model"),
                    default=entry.get("default", False),
                    sha256=entry.get("sha256"),
                    provider=entry.get("provider", "custom"),
                )
            except KeyError:
                continue
        return specs

    def _save_catalog(self) -> None:
        data = [spec.to_dict() for spec in self._custom_specs.values()]
        self._catalog_path.write_text(json.dumps(data, indent=2))

    def available_models(self, include_optional: bool = True) -> List[ModelSpec]:
        specs: Dict[str, ModelSpec] = {}
        for spec in TRUSTED_MODELS:
            specs[spec.filename] = spec
        if include_optional:
            for spec in COMMUNITY_MODELS:
                specs.setdefault(spec.filename, spec)
        for filename, spec in self._custom_specs.items():
            specs[filename] = spec
        return sorted(
            specs.values(), key=lambda spec: (not spec.default, spec.name.lower())
        )

    def custom_models(self) -> List[ModelSpec]:
        return sorted(self._custom_specs.values(), key=lambda spec: spec.name.lower())

    def register_custom_model(self, spec: ModelSpec) -> None:
        with self._lock:
            self._custom_specs[spec.filename] = spec
            self._save_catalog()

    def remove_custom_model(self, filename: str) -> None:
        with self._lock:
            if filename in self._custom_specs:
                del self._custom_specs[filename]
                self._save_catalog()

    # ------------------------------------------------------------------
    # Discovery & download
    # ------------------------------------------------------------------

    def discover(self) -> List[ModelStatus]:
        """Return the current status of all models in the registry."""

        statuses: List[ModelStatus] = []
        for spec in self.available_models(include_optional=True):
            path = self.root / spec.filename
            if path.exists():
                statuses.append(ModelStatus(spec, path, state="available"))
            else:
                message = ""
                if not spec.default:
                    message = "Optional model — enable in Settings to download."
                statuses.append(ModelStatus(spec, None, state="missing", message=message))
        return statuses

    def ensure_default_models(self, *, auto_download: bool = True) -> List[ModelStatus]:
        """Ensure the trusted defaults are present, optionally downloading them."""

        specs = [spec for spec in self.available_models() if spec.default]
        return self.ensure_models(specs, auto_download=auto_download)

    def ensure_models(
        self, specs: Iterable[ModelSpec], *, auto_download: bool = True
    ) -> List[ModelStatus]:
        statuses: List[ModelStatus] = []
        metadata = self._load_metadata()
        skip_flag = os.environ.get("SMALLVILLE_SKIP_MODEL_DOWNLOAD")
        download_enabled = auto_download and not skip_flag and hf_hub_download is not None

        for spec in specs:
            target_path = self.root / spec.filename
            if target_path.exists():
                statuses.append(ModelStatus(spec, target_path, state="available"))
                continue

            if not download_enabled:
                message = metadata.get(spec.filename, {}).get(
                    "message",
                    "Download skipped (offline mode or dependency missing).",
                )
                if hf_hub_download is None:
                    message = "Install huggingface-hub to enable automatic downloads."
                statuses.append(ModelStatus(spec, None, state="missing", message=message))
                continue

            try:
                local_file = Path(
                    hf_hub_download(
                        repo_id=spec.repo_id,
                        filename=spec.filename,
                        local_dir=str(self.root),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                )
                if spec.sha256:
                    self._verify_sha256(local_file, spec.sha256)
                statuses.append(ModelStatus(spec, local_file, state="downloaded"))
            except Exception as exc:  # pragma: no cover - network/IO heavy
                statuses.append(
                    ModelStatus(
                        spec,
                        None,
                        state="error",
                        message=f"Failed to download: {exc}",
                    )
                )

        if statuses:
            with self._lock:
                self._save_metadata(statuses)
        return statuses

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> None:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != expected:
            raise ValueError(
                f"Hash mismatch for {path.name}: expected {expected}, got {digest}"
            )


def ensure_local_models(auto_download: bool = True) -> List[ModelStatus]:
    """Convenience wrapper used by the launcher and GUI."""

    manager = LocalModelManager()
    return manager.ensure_default_models(auto_download=auto_download)


__all__ = [
    "ModelSpec",
    "ModelStatus",
    "TRUSTED_MODELS",
    "COMMUNITY_MODELS",
    "LocalModelManager",
    "ensure_local_models",
]
