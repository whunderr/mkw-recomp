# runtime/shaders/texture

Texture decoding and recompression on the GPU (`gfx/vk/tex/vk_astc.c`).

| File | What it does |
|---|---|
| `astc_decode.comp` | Compute ASTC decoder into RGBA8 |
| `bc7_encode.comp` | Mode 6 BC7 encoder: recompresses the decoded ASTC to 1 byte per texel instead of 4 |
| `bc1_encode.comp` | BC1 encoder: half a byte per texel, for ASTC without alpha |
| `bc3_encode.comp` | BC3 (DXT5) encoder: BC1 colour and interpolated alpha, 1 byte per texel; the alternative to BC7 for ASTC with alpha (Settings > Graphics > Texture compression: BC1 + BC3) |
| `bc2_encode.comp` | BC2 (DXT3) encoder: BC1 colour and 4-bit alpha, 1 byte per texel; the same, as BC1 + BC2 |
| `bcn_rgba.glsl` | The body `bc3_encode.comp` and `bc2_encode.comp` share (`#include`d after each sets `BC_ALPHA_MODE`) |
