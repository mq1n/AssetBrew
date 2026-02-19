from __future__ import annotations

from pathlib import Path

import pytest

from AssetBrew.ui.core import (
    find_output_map_file,
    parse_typed_value,
    resolve_map_paths,
    set_dataclass_path,
    value_to_text,
)


def test_value_to_text_boolean_and_collections() -> None:
    assert value_to_text(None) == "null"
    assert value_to_text(True) == "true"
    assert value_to_text(False) == "false"
    assert value_to_text([1, 2, 3]).startswith("[")
    assert value_to_text({"a": 1}).startswith("{")


def test_parse_typed_value_basic_types() -> None:
    assert parse_typed_value("12", 1) == 12
    assert parse_typed_value("3.5", 1.0) == 3.5
    assert parse_typed_value("YES", False) is True
    assert parse_typed_value("no", True) is False
    assert parse_typed_value(" hello ", "") == " hello "
    assert parse_typed_value("None", None) is None
    assert parse_typed_value("none", None) is None
    assert parse_typed_value("null", None) is None
    assert parse_typed_value("0.7", None) == 0.7


def test_parse_typed_value_list_and_dict() -> None:
    assert parse_typed_value("[1, 2, 3]", []) == [1, 2, 3]
    assert parse_typed_value("{a: 1, b: 2}", {}) == {"a": 1, "b": 2}


def test_parse_typed_value_raises_on_invalid_boolean() -> None:
    with pytest.raises(ValueError):
        parse_typed_value("not-bool", True)


def test_set_dataclass_path_updates_nested_attr() -> None:
    class Nested:
        def __init__(self) -> None:
            self.value = 1

    class Root:
        def __init__(self) -> None:
            self.nested = Nested()

    root = Root()
    set_dataclass_path(root, ("nested", "value"), 99)
    assert root.nested.value == 99


def test_find_output_map_file_uses_extension_priority(monkeypatch) -> None:
    output_dir = "C:/out"
    png = "C:/out/hero/brick_diff_normal.png"
    dds = "C:/out/hero/brick_diff_normal.dds"
    existing = {png, dds}

    def fake_exists(self) -> bool:  # noqa: ANN001
        return str(self).replace("\\", "/") in existing

    monkeypatch.setattr(Path, "exists", fake_exists)

    found = find_output_map_file(
        rel_path="hero/brick_diff.png",
        output_dir=output_dir,
        suffix="_normal",
        ext_priority=[".png", ".dds"],
    )
    assert found.replace("\\", "/") == png


def test_resolve_map_paths_prefers_result_entries(monkeypatch) -> None:
    output_dir = "C:/out"
    fallback_normal = "C:/out/hero/brick_diff_normal.png"
    direct_normal = "C:/tmp/direct_normal.png"
    existing = {fallback_normal, direct_normal}

    def fake_exists(self) -> bool:  # noqa: ANN001
        return str(self).replace("\\", "/") in existing

    monkeypatch.setattr(Path, "exists", fake_exists)

    paths = resolve_map_paths(
        rel_path="hero/brick_diff.png",
        result_entry={"normal": {"normal": direct_normal}},
        output_dir=output_dir,
    )
    assert paths["Normal"].replace("\\", "/") == direct_normal


def test_resolve_map_paths_falls_back_to_output_dir(monkeypatch) -> None:
    output_dir = "C:/out"
    base = "C:/out/hero/brick_diff.png"
    rough = "C:/out/hero/brick_diff_roughness.png"
    existing = {base, rough}

    def fake_exists(self) -> bool:  # noqa: ANN001
        return str(self).replace("\\", "/") in existing

    monkeypatch.setattr(Path, "exists", fake_exists)

    paths = resolve_map_paths(
        rel_path="hero/brick_diff.png",
        result_entry={},
        output_dir=output_dir,
    )
    assert paths["Base"].replace("\\", "/") == base
    assert paths["Roughness"].replace("\\", "/") == rough


def test_resolve_map_paths_prefers_graded_and_new_maps(monkeypatch) -> None:
    output_dir = "C:/out"
    graded = "C:/tmp/graded.png"
    gloss = "C:/tmp/gloss.png"
    emissive = "C:/tmp/emissive.png"
    envmask = "C:/tmp/envmask.png"
    zonemask = "C:/tmp/zonemask.png"
    existing = {graded, gloss, emissive, envmask, zonemask}

    def fake_exists(self) -> bool:  # noqa: ANN001
        return str(self).replace("\\", "/") in existing

    monkeypatch.setattr(Path, "exists", fake_exists)

    paths = resolve_map_paths(
        rel_path="hero/brick_diff.png",
        result_entry={
            "color_grading": {"graded": graded},
            "pbr": {"gloss": gloss, "zone_mask": zonemask},
            "emissive": {"emissive": emissive},
            "reflection_mask": {"env_mask": envmask},
        },
        output_dir=output_dir,
    )
    assert paths["Albedo"].replace("\\", "/") == graded
    assert paths["Gloss"].replace("\\", "/") == gloss
    assert paths["Emissive"].replace("\\", "/") == emissive
    assert paths["Env Mask"].replace("\\", "/") == envmask
    assert paths["Zone Mask"].replace("\\", "/") == zonemask


def test_resolve_map_paths_base_prefers_diffuse_alpha_packed(monkeypatch) -> None:
    output_dir = "C:/out"
    packed = "C:/tmp/base_packed.png"
    seam = "C:/tmp/base_seam.png"
    upscaled = "C:/tmp/base_upscaled.png"
    existing = {packed, seam, upscaled}

    def fake_exists(self) -> bool:  # noqa: ANN001
        return str(self).replace("\\", "/") in existing

    monkeypatch.setattr(Path, "exists", fake_exists)

    paths = resolve_map_paths(
        rel_path="hero/brick_diff.png",
        result_entry={
            "orm": {"diffuse_alpha_packed": packed},
            "seam_repair": {"upscaled_repaired": seam},
            "upscale": {"upscaled": upscaled},
        },
        output_dir=output_dir,
    )
    assert paths["Base"].replace("\\", "/") == packed
