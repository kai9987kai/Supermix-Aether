"""SHA256 artifact manifests detect changes; they are not authenticity signatures.

The source fingerprint covers a fixed source/config/documentation allowlist,
not run directories. Receipt verification does not execute referenced content.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys
import zipfile


_RECEIPT = "receipt.json"
_SOURCE_DIRS = {"aether", "configs", "docs", "tests"}
_SOURCE_ROOT_FILES = {"pyproject.toml", "README.md", "requirements.txt", "requirements-dev.txt",
                      "uv.lock", "poetry.lock"}
_SOURCE_SUFFIXES = {".py", ".json", ".toml", ".md", ".yaml", ".yml", ".js", ".css", ".html"}
_EXCLUDED = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git",
             "node_modules", ".venv", "venv", "runs", "outputs", "reports", "dist", "build"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                       object_pairs_hook=_unique_object)
    # JSON exponents such as 1e400 overflow without invoking parse_constant.
    _canonical(value)
    return value


def _canonical(value):
    # Reparse to reject unusual JSON values and non-string mapping keys before
    # a receipt can be written. allow_nan=False also catches nested nonfinites.
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)


def _artifact_paths(root):
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for entry in dirs + files:
            path = Path(directory) / entry
            if path.is_symlink() or path.is_junction():
                raise ValueError(f"linked artifact paths are not supported: {path.relative_to(root)}")
        for entry in files:
            path = Path(directory) / entry
            if path == root / _RECEIPT:
                continue
            if not path.is_file():
                raise ValueError(f"artifact is not a regular file: {path}")
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _source_paths(root):
    """One source allowlist shared by fingerprints and preserved snapshots."""
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED
                         and not (current / name).is_symlink()
                         and not (current / name).is_junction()
                         and (current != root or name in _SOURCE_DIRS))
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(root)
            if path.is_symlink():
                continue
            if ((current == root and name in _SOURCE_ROOT_FILES)
                    or (relative.parts[0] in _SOURCE_DIRS and path.suffix in _SOURCE_SUFFIXES)):
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def source_fingerprint(root):
    """Stable SHA256 over relative paths and bytes in the source allowlist.

    Includes aether/, configs/, docs/, tests/ and explicit project root files.
    Excludes generated outputs, caches, build directories and linked paths.
    """
    root = Path(root).resolve(strict=True)
    manifest = [{"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
                for path in _source_paths(root)]
    return hashlib.sha256(_canonical(sorted(manifest, key=lambda x: x["path"])).encode("utf-8")).hexdigest()


def write_source_snapshot(root, destination):
    """Preserve the complete fingerprint allowlist and return its fingerprint.

    Hash the exact bytes written to the archive, so provenance describes the
    preserved source even if a source file changes while the study is starting.
    The archive is created exclusively and never replaces an existing snapshot.
    """
    root = Path(root).resolve(strict=True)
    manifest = []
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _source_paths(root):
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            archive.writestr(relative, content)
            manifest.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    return hashlib.sha256(_canonical(manifest).encode('utf-8')).hexdigest()


def _environment():
    packages = {}
    for name in ("numpy", "scipy", "torch", "aether"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages}


def write_receipt(output_dir, metadata=None):
    """Create receipt.json exclusively after output artifacts are finalized.

    All regular output files are hashed. Existing receipts are never replaced.
    Metadata may supply a source_fingerprint and environment; otherwise these
    describe this installed implementation and its current runtime. Record a
    project fingerprint explicitly when running from another source checkout.
    """
    root = Path(output_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("output_dir must be a directory")
    receipt = root / _RECEIPT
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(receipt)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON-safe dictionary")
    # Serialize now, before any mutation, and detach caller-owned structures.
    metadata = json.loads(_canonical(metadata), parse_constant=_reject_constant)
    files = []
    for path in _artifact_paths(root):
        if path.suffix.lower() == ".json":
            _load_json(path)
        files.append({"path": path.relative_to(root).as_posix(),
                      "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    if not files:
        raise ValueError("cannot create an empty artifact manifest")
    fingerprint = metadata.get("source_fingerprint", source_fingerprint(Path(__file__).resolve().parents[2]))
    if not isinstance(fingerprint, str) or not _DIGEST.fullmatch(fingerprint):
        raise ValueError("source_fingerprint must be a lowercase SHA256 digest")
    environment = metadata.get("environment", _environment())
    if not isinstance(environment, dict) or not environment:
        raise ValueError("environment must be a nonempty JSON object")
    payload = {"schema_version": 1, "hash_algorithm": "sha256", "files": files,
               "source_fingerprint": fingerprint, "environment": environment,
               "metadata": metadata,
               "scope": "change detection only; unsigned and not proof of authenticity"}
    content = _canonical(payload) + "\n"
    with receipt.open("x", encoding="utf-8", newline="\n") as target:
        target.write(content)
    return receipt


def _safe_reference(root, reference):
    if (not isinstance(reference, str) or not reference or "\\" in reference
            or ":" in reference or "\x00" in reference):
        raise ValueError("unsafe artifact reference")
    relative = PurePosixPath(reference)
    if (relative.is_absolute() or PureWindowsPath(reference).drive
            or any(part in ("..", ".", "") for part in reference.split("/"))):
        raise ValueError(f"unsafe artifact reference: {reference}")
    path = root.joinpath(*relative.parts)
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError(f"linked artifact reference: {reference}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"artifact escapes receipt root: {reference}")
    return path


def verify_receipt(path):
    """Return a JSON-safe validity/details report; malformed evidence fails closed."""
    details = []
    checked = 0
    try:
        receipt = Path(path).absolute()
        if receipt.is_symlink() or receipt.is_junction():
            raise ValueError("receipt itself must not be a link")
        root = receipt.parent.resolve(strict=True)
        payload = _load_json(receipt)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
            raise ValueError("unsupported receipt schema")
        _canonical(payload)
        if not isinstance(payload.get("source_fingerprint"), str) or not _DIGEST.fullmatch(payload["source_fingerprint"]):
            raise ValueError("missing or invalid source fingerprint")
        if not isinstance(payload.get("environment"), dict) or not payload["environment"]:
            raise ValueError("missing environment metadata")
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("empty or invalid artifact manifest")
        seen = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("invalid manifest entry")
            reference = item.get("path")
            artifact = _safe_reference(root, reference)
            if reference == _RECEIPT or reference in seen:
                raise ValueError("duplicate or self-referencing artifact")
            seen.add(reference)
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or type(size) is not int or size < 0:
                raise ValueError(f"invalid hash or size for {reference}")
            if not artifact.is_file():
                details.append(f"missing artifact: {reference}")
                continue
            if artifact.stat().st_size != size or _sha256(artifact) != digest:
                details.append(f"changed artifact: {reference}")
            if artifact.suffix.lower() == ".json":
                _load_json(artifact)
            checked += 1
        actual = {p.relative_to(root).as_posix() for p in _artifact_paths(root)}
        for reference in sorted(actual - seen):
            details.append(f"unlisted artifact: {reference}")
    except (ValueError, OSError, TypeError, OverflowError, RecursionError) as exc:
        details.append(f"invalid receipt: {exc}")
    return {"valid": not details, "checked_files": checked, "details": details,
            "scope": "change detection only; unsigned and not proof of authenticity"}
