"""Asset record dataclass."""

from dataclasses import dataclass, asdict
from pathlib import PurePosixPath, PureWindowsPath


@dataclass
class AssetRecord:
    """Single asset entry in the manifest."""

    filepath: str
    filename: str
    texture_type: str
    original_width: int
    original_height: int
    channels: int
    has_alpha: bool
    is_tileable: bool
    is_hero: bool
    material_category: str
    file_size_kb: float
    file_hash: str = ""
    is_gloss: bool = False
    status: str = "pending"

    def __post_init__(self) -> None:
        """Normalize and validate stored relative filepath for stable matching."""
        original = str(self.filepath)
        raw = original.replace("\\", "/")
        p = PurePosixPath(raw)
        win = PureWindowsPath(original)
        drive_like = bool(raw) and len(raw) >= 2 and raw[1] == ":"
        unc_like = raw.startswith("//")
        if p.is_absolute() or win.is_absolute() or drive_like or unc_like:
            raise ValueError(f"AssetRecord.filepath must be relative, got: {self.filepath}")

        parts = []
        for part in p.parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    raise ValueError(
                        f"AssetRecord.filepath escapes root via '..': {self.filepath}"
                    )
                parts.pop()
                continue
            parts.append(part)

        if not parts:
            raise ValueError(f"AssetRecord.filepath is empty after normalization: {self.filepath}")
        self.filepath = "/".join(parts)

    def to_dict(self) -> dict:
        """Return dataclass fields as a plain dictionary."""
        return asdict(self)
