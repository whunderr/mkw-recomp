"""Settings > Graphics > Texture compression (MKW_TEX_COMPRESSION): the values the settings table, the reader in
gfx/vk/tex/vk_astc.c and the menu row agree on, and the embedded SPIR-V of the BC3 and BC2 encoders is the one their
sources compile to. Nothing here needs a GPU or the game; the encoders' output was checked on a software Vulkan device
against an independent BC2/BC3 decoder when they were added (runtime/shaders/texture/README.md).

Usage: uv run pytest tests
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = (ROOT / "runtime/core/base/cfg.c").read_text()
ASTC = (ROOT / "runtime/gfx/vk/tex/vk_astc.c").read_text()
MENU = (ROOT / "runtime/ui/menu/settings_tabs.cpp").read_text()


def row() -> tuple[str, list[str]]:
    m = re.search(r'S\("TEX_COMPRESSION", "graphics", "([^"]*)", "([^"]*)"', CFG)
    assert m, "no TEX_COMPRESSION row in cfg.c"
    return m.group(1), m.group(2).split("|")


def test_default_is_the_current_behaviour():
    default, values = row()
    assert default == "auto"  # BC1 for opaque ASTC, BC7 for the rest: what the port did before the setting
    assert default in values


def test_reader_handles_every_value():
    _, values = row()
    for v in values:
        if v != "auto":  # auto is the fall-through
            assert f'!strcmp(e, "{v}")' in ASTC, v


def test_menu_offers_every_value():
    _, values = row()
    block = MENU[MENU.index('vals[] = {"auto", "bc3"') :][:200]
    for v in values:
        assert f'"{v}"' in block, v


def test_bc3_bc2_spirv_is_current():
    for name in ("bc3_encode", "bc2_encode"):
        src = ROOT / f"runtime/shaders/texture/{name}.comp"
        hdr = (ROOT / f"runtime/gfx/vk/spv/{name}_spv.h").read_text()
        assert hashlib.sha256(src.read_bytes()).hexdigest() in hdr, f"{name}: run tools/shaders/embed_spv.py"
        assert (ROOT / "runtime/shaders/texture/bcn_rgba.glsl").exists()
