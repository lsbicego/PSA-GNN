"""Ordinary per-graph PyG files, local loading, and optional Hub archives."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage

FORMAT = "psa-gnn-per-graph"
FORMAT_VERSION = 1
_NAMES = {"macro": "macro-hierarchies", "macro-hierarchies": "macro-hierarchies",
          "granular": "granular-hierarchies", "granular-hierarchies": "granular-hierarchies"}
_SAFE_PYG = [Data, DataEdgeAttr, DataTensorAttr, BaseStorage, EdgeStorage, GlobalStorage, NodeStorage]


def _name(name: str) -> str:
    try:
        return _NAMES[name.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError("name must be 'macro' or 'granular' (or their full names).") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path) -> tuple[Path, list[Path]]:
    raw = root / "raw" if (root / "raw").is_dir() else root
    if not raw.is_dir():
        raise FileNotFoundError(f"No dataset directory at {root}; generate data or download/extract an archive first.")
    pairs = [(int(match.group(1)), path) for path in raw.iterdir()
             if path.is_file() and (match := re.fullmatch(r"graph_(\d+)\.pt", path.name))]
    pairs.sort(key=lambda pair: pair[0])
    if not pairs:
        raise FileNotFoundError(f"No graph_N.pt files in {raw}; pass the dataset root or its raw directory.")
    if len({index for index, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate numeric graph indices in dataset filenames.")
    return raw, [path for _, path in pairs]


def _load_graph(path: Path, trusted: bool = False) -> Data:
    try:
        if trusted:
            data = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with torch.serialization.safe_globals(_SAFE_PYG):
                data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot load {path} with the standard PyG allowlist. Custom Python objects require trusted=True for explicitly trusted local files only.") from exc
    if not isinstance(data, Data):
        raise ValueError(f"{path} does not contain a PyG Data graph.")
    if data.num_nodes is None or data.num_nodes < 0:
        raise ValueError(f"{path} has no valid node count.")
    if data.edge_index is None or data.edge_index.dtype != torch.long or data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(f"{path}: edge_index must be a long [2, E] tensor.")
    if data.edge_index.numel() and (int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= data.num_nodes):
        raise ValueError(f"{path}: an edge index is outside the graph.")
    if data.x is None or data.x.ndim != 2 or data.x.shape[0] != data.num_nodes or data.y is None or data.y.numel() != 1:
        raise ValueError(f"{path}: expected node features x and one graph label y.")
    return data


def _new_directory(root: Path) -> None:
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output {root} is not empty; choose a fresh directory.")
    root.mkdir(parents=True, exist_ok=True)


def _manifest(files: list[Path], *, name: str, generation=None, split_indices=None, metadata=None) -> dict:
    splits = None if split_indices is None else {key: list(value) for key, value in split_indices.items()}
    flat = [] if splits is None else [index for values in splits.values() for index in values]
    if any(type(index) is not int or not 0 <= index < len(files) for index in flat) or len(flat) != len(set(flat)):
        raise ValueError("Split indices must be valid, nonoverlapping graph indices.")
    result = {"format": FORMAT, "format_version": FORMAT_VERSION, "name": _name(name),
              "num_graphs": len(files),
              "files": [{"path": f"raw/{path.name}", "sha256": _sha256(path)} for path in files],
              "ordering": "Numeric graph_N filename order; historical filesystem enumeration is not inferred.",
              "generation": dict(generation or {}),
              "splits": {"indices": splits, "provenance": "provided" if splits is not None else "No canonical split supplied."},
              "software": {package: version(package) for package in ("torch", "torch-geometric", "numpy", "networkx")}}
    if metadata is not None:
        if len(metadata) != len(files):
            raise ValueError("metadata must have one entry per graph.")
        result["graph_metadata"] = list(metadata)
    json.dumps(result, allow_nan=False)
    return result


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def save_synthetic_dataset(graphs: Iterable[Data], root: str | Path, *, name: str,
                           metadata: Sequence[Mapping[str, Any]] | None = None,
                           generation: Mapping[str, Any] | None = None,
                           split_indices: Mapping[str, Sequence[int]] | None = None) -> Path:
    """Save separate PyG objects in a fresh root/raw directory with a manifest."""
    root = Path(root).expanduser()
    _name(name)
    graphs = list(graphs)
    if not graphs or any(not isinstance(graph, Data) for graph in graphs):
        raise ValueError("Supply at least one PyG Data graph.")
    _new_directory(root)
    (root / "raw").mkdir()
    files = []
    for index, graph in enumerate(graphs):
        path = root / "raw" / f"graph_{index}.pt"
        torch.save(graph, path)
        files.append(path)
    _write_manifest(root / "manifest.json", _manifest(files, name=name, metadata=metadata,
                    generation=generation, split_indices=split_indices))
    return root


class SyntheticHierarchicalData(Dataset):
    """Read original graph_N.pt objects on demand; a manifest is optional.

    Accepts a dataset root containing raw/ or the raw directory itself. Standard
    PyG objects load with a restricted allowlist; trusted=True is only for local
    custom objects. Preprocessing is cached separately as per-graph files.
    """

    def __init__(self, root: str | Path, *, name: str | None = None, transform=None,
                 pre_transform=None, pre_filter=None, processed_root: str | Path | None = None,
                 force_reload: bool = False, verify: bool = True, trusted: bool = False):
        source = Path(root).expanduser().resolve()
        raw, paths = _files(source)
        base = raw.parent if raw.name == "raw" else source
        manifest_path = base / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if manifest and (manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION):
            raise ValueError("Unsupported manifest format/version; expected original per-graph PSA-GNN data.")
        if manifest:
            entries = manifest.get("files", [])
            if [entry.get("path") for entry in entries] != [f"raw/{path.name}" for path in paths] or manifest.get("num_graphs") != len(paths):
                raise ValueError("Manifest graph list does not match raw/graph_N.pt files.")
            if verify and any(_sha256(path) != entry.get("sha256") for path, entry in zip(paths, entries)):
                raise ValueError("Checksum mismatch in graph files; restore the original dataset.")
        self.manifest = manifest
        self.name = manifest.get("name", _name(name) if name is not None else "synthetic-hierarchies")
        self.graph_metadata = manifest.get("graph_metadata", [{} for _ in paths])
        self.split_indices = manifest.get("splits", {}).get("indices")
        self.source_indices = list(range(len(paths)))
        self.trusted = trusted
        super().__init__(root=str(base), transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, log=False)
        if pre_transform is not None or pre_filter is not None:
            identity = str(base) + json.dumps(manifest, sort_keys=True) + repr(pre_transform) + repr(pre_filter)
            identity += repr([(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in paths])
            cache = (Path(processed_root).expanduser() if processed_root is not None else base / "processed_public") / hashlib.sha256(identity.encode()).hexdigest()
            index_file = cache / "index.json"
            if index_file.is_file() and not force_reload:
                self.source_indices = json.loads(index_file.read_text())["source_indices"]
            else:
                cache.mkdir(parents=True, exist_ok=True)
                kept = []
                for index, path in enumerate(paths):
                    data = _load_graph(path, trusted)
                    if pre_filter is not None and not pre_filter(data):
                        continue
                    data = pre_transform(data) if pre_transform is not None else data
                    torch.save(data, cache / f"graph_{len(kept)}.pt")
                    kept.append(index)
                self.source_indices = kept
                _write_manifest(index_file, {"source_indices": kept})
            paths = [cache / f"graph_{index}.pt" for index in range(len(self.source_indices))]
            self.graph_metadata = [self.graph_metadata[index] for index in self.source_indices]
            if self.split_indices is not None:
                lookup = {old: new for new, old in enumerate(self.source_indices)}
                self.split_indices = {key: [lookup[index] for index in values if index in lookup] for key, values in self.split_indices.items()}
        self._paths = paths

    def len(self) -> int:
        return len(self._paths)

    def get(self, index: int) -> Data:
        return _load_graph(self._paths[index], self.trusted)

    @classmethod
    def from_huggingface(cls, repo_id: str, *, subfolder: str = "", revision: str | None = None,
                         cache_dir: str | Path | None = None, local_files_only: bool = False,
                         **dataset_kwargs) -> "SyntheticHierarchicalData":
        """Cache a manifest + graphs.zip variant and load its original graph files."""
        folder = PurePosixPath(subfolder)
        if folder.is_absolute() or ".." in folder.parts or "\\" in subfolder:
            raise ValueError("subfolder must be a relative repository path.")
        if dataset_kwargs.get("trusted") or dataset_kwargs.get("verify", True) is False:
            raise ValueError("Hub datasets require restricted loading and checksum verification.")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install requirements.txt for Hugging Face support, or load a local directory.") from exc
        args = dict(repo_id=repo_id, repo_type="dataset", revision=revision,
                    cache_dir=None if cache_dir is None else str(cache_dir), local_files_only=local_files_only)
        try:
            manifest_file = Path(hf_hub_download(filename=str(folder / "manifest.json"), **args))
            archive = Path(hf_hub_download(filename=str(folder / "graphs.zip"), **args))
        except Exception as exc:
            raise OSError(f"Cannot download {repo_id!r}; check repository/revision or already cached files for offline use.") from exc
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported archive manifest format/version.")
        if _sha256(archive) != manifest.get("archive_sha256"):
            raise ValueError("Archive checksum mismatch; restore the published archive.")
        cache_base = Path(cache_dir).expanduser() if cache_dir is not None else Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        extracted = cache_base / "psa-gnn-extracted" / hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        # Derived preprocessing/extraction must not modify immutable snapshots.
        processed = dataset_kwargs.get("processed_root")
        for downloaded in (manifest_file, archive):
            protected = downloaded.parent.resolve()
            for ancestor in (protected, *protected.parents):
                if ancestor.parent.name == "snapshots":
                    protected = ancestor
                    break
            for directory in (extracted, None if processed is None else Path(processed).expanduser()):
                if directory is not None and (directory.resolve() == protected or protected in directory.resolve().parents):
                    raise ValueError("Extraction and processed_root must stay outside the downloaded Hub snapshot.")
        if not (extracted / "manifest.json").is_file():
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="extract-", dir=extracted.parent) as temporary:
                destination = Path(temporary)
                entries = manifest.get("files", [])
                expected = {entry["path"] for entry in entries}
                if len(expected) != len(entries) or not expected:
                    raise ValueError("Archive manifest needs a unique, nonempty graph file list.")
                with zipfile.ZipFile(archive) as bundle:
                    members = bundle.infolist()
                    graph_members = [member for member in members if not member.is_dir()]
                    if len(graph_members) != len(expected) or {member.filename for member in graph_members} != expected:
                        raise ValueError("Archive contents differ from the manifest file list.")
                    for member in members:
                        path = PurePosixPath(member.filename)
                        allowed_name = member.filename == "raw/" if member.is_dir() else re.fullmatch(r"raw/graph_\d+\.pt", member.filename)
                        if not allowed_name or path.is_absolute() or ".." in path.parts or stat.S_ISLNK(member.external_attr >> 16):
                            raise ValueError("Unsafe archive member path or symlink.")
                        if member.is_dir():
                            continue
                        output = destination / member.filename
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.open(member) as incoming, output.open("wb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                for entry in entries:
                    if _sha256(destination / entry["path"]) != entry.get("sha256"):
                        raise ValueError("Extracted graph checksum mismatch.")
                shutil.copy2(manifest_file, destination / "manifest.json")
                if extracted.exists():
                    raise FileExistsError(f"Incomplete extraction cache at {extracted}; choose another cache_dir.")
                destination.rename(extracted)
        result = cls(extracted, **dataset_kwargs)
        result.hub_source = {"repo_id": repo_id, "revision": revision, "subfolder": subfolder}
        return result


def generate_synthetic_dataset(root: str | Path, *, name: str = "macro", num_graphs: int = 100,
                               seed: int = 0, **generator_kwargs) -> SyntheticHierarchicalData:
    """Call the unchanged research generator and its original per-graph saver."""
    from src.data_generation import synthetic_datasets as original
    canonical = _name(name)
    if type(num_graphs) is not int or num_graphs < 1:
        raise ValueError("num_graphs must be a positive integer.")
    root = Path(root).expanduser()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Output {root} is not empty; choose a fresh directory.")
    generator = original.generate_hsbm_macro_topology_variable_dataset if canonical == "macro-hierarchies" else original.generate_degree_matched_granularity_dataset_v2
    bound = inspect.signature(generator).bind(num_graphs=num_graphs, seed=seed, **generator_kwargs)
    bound.apply_defaults()
    graphs, metadata = generator(**bound.arguments)
    original.save_graph_dataset_to_disk(graphs, metadata, str(root / "raw"), show_progress=False)
    _, paths = _files(root)
    manifest = _manifest(paths, name=canonical, metadata=metadata,
                         generation={"function": generator.__name__, "parameters": bound.arguments,
                                     "generator_sha256": _sha256(Path(original.__file__)),
                                     "features": "original saver defaults; constant ones"})
    _write_manifest(root / "manifest.json", manifest)
    return SyntheticHierarchicalData(root)


def pack_synthetic_dataset(source: str | Path, output: str | Path, *, name: str,
                           generation: Mapping[str, Any] | None = None,
                           split_indices: Mapping[str, Sequence[int]] | None = None) -> Path:
    """Archive existing graph files byte-for-byte; no deserialization or conversion."""
    source = Path(source).expanduser()
    raw, paths = _files(source)
    base = raw.parent if raw.name == "raw" else source
    previous = json.loads((base / "manifest.json").read_text()) if (base / "manifest.json").is_file() else {}
    manifest = _manifest(paths, name=name, generation=generation if generation is not None else previous.get("generation"),
                         split_indices=split_indices if split_indices is not None else previous.get("splits", {}).get("indices"),
                         metadata=previous.get("graph_metadata"))
    output = Path(output).expanduser()
    _new_directory(output)
    with zipfile.ZipFile(output / "graphs.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=f"raw/{path.name}")
    manifest["archive_sha256"] = _sha256(output / "graphs.zip")
    _write_manifest(output / "manifest.json", manifest)
    return output
