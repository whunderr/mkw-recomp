// Shared body of bc3_encode.comp (BC_ALPHA_MODE 3) and bc2_encode.comp (BC_ALPHA_MODE 2): RGBA8 texels -> 16-byte
// BC3 (DXT5) or BC2 (DXT3) blocks. Used after the ASTC compute decoder (vk_astc.c) for textures with alpha when the
// Texture compression setting (MKW_TEX_COMPRESSION) asks for BC3 or BC2 where it would otherwise use BC7. Same size as
// BC7 (1 byte a texel); the point is a cheaper encode and a format every GPU with BC support decodes the same way.
//   Colour: the BC1 fit of bc1_encode.comp (principal axis, one least-squares refinement), four-colour mode, which BC2
//           and BC3 always use whatever the endpoint order.
//   BC2 alpha: 4 bits a texel, the nearest of 0, 17, 34 ... 255.
//   BC3 alpha: two 8-bit endpoints and 3-bit indices. Two palettes are tried and the smaller squared error wins:
//           (A) a0 = max > a1 = min, six values between them (an exact 255 or 0 is kept when it is an endpoint);
//           (B) a0 <= a1 over the texels that are neither 0 nor 255, four values between, plus an exact 0 and 255, for
//           the blocks of a cut-out's edge. Alpha 0 and 255 are exact in both.
// Bit layout: Khronos Data Format Specification 1.3, BC2 and BC3 sections. The block is the alpha half first
// (uvec4.xy), then the BC1 colour half (uvec4.zw).
#ifndef BC_ALPHA_MODE
#error "BC_ALPHA_MODE must be 2 (BC2) or 3 (BC3)"
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer Texels { uint px[]; };    // RGBA8, R in the low byte
layout(std430, set = 0, binding = 1) writeonly buffer Blocks { uvec4 blk[]; };

layout(push_constant) uniform Params {
    uint in_base;    // first texel of the region (uint index)
    uint out_base;   // first block of the region (uvec4 index)
    uint width, height;
    uint bx, bslice;   // blocks per row, blocks per slice
};

vec4 texel(uint x, uint y, uint z) {   // edge texels repeat past the image (blocks of a small mip)
    x = min(x, width - 1u), y = min(y, height - 1u);
    uint v = px[in_base + (z * height + y) * width + x];
    return vec4(v & 255u, (v >> 8) & 255u, (v >> 16) & 255u, v >> 24);
}

uint pack565(vec3 c) {
    uvec3 q = uvec3(clamp(round(c * vec3(31.0, 63.0, 31.0) / 255.0), vec3(0.0), vec3(31.0, 63.0, 31.0)));
    return (q.r << 11) | (q.g << 5) | q.b;
}

vec3 unpack565(uint v) {
    uvec3 q = uvec3(v >> 11, (v >> 5) & 63u, v & 31u);
    return vec3(q.r * 255u / 31u, q.g * 255u / 63u, q.b * 255u / 31u);   // as BC1 decoders expand (bit replication)
}

// Indices of the four palette colours nearest to each texel, and their summed error.
float fit(vec3 c[16], uint e0, uint e1, out uint idx[16]) {
    vec3 a = unpack565(e0), b = unpack565(e1);
    vec3 pal[4] = vec3[](a, b, (2.0 * a + b) / 3.0, (a + 2.0 * b) / 3.0);
    float total = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        float best = 1e9;
        uint bi = 0u;
        for (uint k = 0u; k < 4u; k++) {
            vec3 d = pal[k] - c[i];
            float err = dot(d, d);
            if (err < best) best = err, bi = k;
        }
        idx[i] = bi;
        total += best;
    }
    return total;
}

void order(inout uint e0, inout uint e1) {
    if (e0 < e1) { uint t = e0; e0 = e1; e1 = t; }
}

// The colour half: two uints, endpoints then 2-bit indices (bc1_encode.comp).
uvec2 encode_colour(vec3 c[16]) {
    vec3 mean = vec3(0.0);
    for (uint i = 0u; i < 16u; i++) mean += c[i];
    mean *= 1.0 / 16.0;
    mat3 cov = mat3(0.0);
    for (uint i = 0u; i < 16u; i++) {
        vec3 d = c[i] - mean;
        cov += outerProduct(d, d);
    }
    vec3 axis = vec3(1.0, 1.0, 1.0);
    for (int k = 0; k < 6; k++) {
        axis = cov * axis;
        float l = length(axis);
        axis = l > 1e-6 ? axis / l : vec3(0.577);
    }
    float tmin = 1e9, tmax = -1e9;
    for (uint i = 0u; i < 16u; i++) {
        float t = dot(c[i] - mean, axis);
        tmin = min(tmin, t), tmax = max(tmax, t);
    }
    uint e0 = pack565(clamp(mean + axis * tmax, 0.0, 255.0)), e1 = pack565(clamp(mean + axis * tmin, 0.0, 255.0));
    order(e0, e1);
    uint idx[16];
    float err = fit(c, e0, e1, idx);
    const float W[4] = float[](1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0);
    float aa = 0.0, bb = 0.0, ab = 0.0;
    vec3 ax = vec3(0.0), bx_ = vec3(0.0);
    for (uint i = 0u; i < 16u; i++) {
        float w = W[idx[i]];
        aa += w * w, bb += (1.0 - w) * (1.0 - w), ab += w * (1.0 - w);
        ax += w * c[i], bx_ += (1.0 - w) * c[i];
    }
    float det = aa * bb - ab * ab;
    if (abs(det) > 1e-4 && e0 != e1) {
        vec3 r0 = (ax * bb - bx_ * ab) / det, r1 = (bx_ * aa - ax * ab) / det;
        uint f0 = pack565(clamp(r0, 0.0, 255.0)), f1 = pack565(clamp(r1, 0.0, 255.0));
        order(f0, f1);
        uint idx2[16];
        float err2 = f0 != f1 ? fit(c, f0, f1, idx2) : 1e9;
        if (err2 < err) {
            e0 = f0, e1 = f1;
            for (uint i = 0u; i < 16u; i++) idx[i] = idx2[i];
        }
    }
    uint bits = 0u;
    if (e0 != e1)
        for (uint i = 0u; i < 16u; i++) bits |= idx[i] << (2u * i);
    return uvec2(e0 | (e1 << 16), bits);
}

#if BC_ALPHA_MODE == 2

// BC2: texel i (row-major) in bits 4i.. of the 64-bit alpha half.
uvec2 encode_alpha(float al[16]) {
    uvec2 r = uvec2(0u);
    for (uint i = 0u; i < 16u; i++) r[i >> 3] |= ((uint(al[i]) + 8u) / 17u) << (4u * (i & 7u));
    return r;
}

#else

// The eight palette values of the endpoints a0, a1 (a0 > a1: six between; otherwise four between, then 0 and 255).
void palette(uint a0, uint a1, out float p[8]) {
    float f0 = float(a0), f1 = float(a1);
    p[0] = f0, p[1] = f1;
    if (a0 > a1) {
        for (uint k = 1u; k <= 6u; k++) p[1u + k] = (float(7u - k) * f0 + float(k) * f1) / 7.0;
    } else {
        for (uint k = 1u; k <= 4u; k++) p[1u + k] = (float(5u - k) * f0 + float(k) * f1) / 5.0;
        p[6] = 0.0, p[7] = 255.0;
    }
}

float alpha_fit(float al[16], float p[8], out uint idx[16]) {
    float total = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        float best = 1e9;
        uint bi = 0u;
        for (uint k = 0u; k < 8u; k++) {
            float d = p[k] - al[i];
            if (d * d < best) best = d * d, bi = k;
        }
        idx[i] = bi;
        total += best;
    }
    return total;
}

// BC3: byte 0 = a0, byte 1 = a1, then 16 3-bit indices from bit 16 of the 64-bit alpha half.
uvec2 encode_alpha(float al[16]) {
    float amin = 255.0, amax = 0.0, mid_min = 255.0, mid_max = 0.0;
    bool has_mid = false, has_end = false;
    for (uint i = 0u; i < 16u; i++) {
        amin = min(amin, al[i]), amax = max(amax, al[i]);
        if (al[i] > 0.5 && al[i] < 254.5) has_mid = true, mid_min = min(mid_min, al[i]), mid_max = max(mid_max, al[i]);
        else has_end = true;
    }
    uint a0 = uint(amax), a1 = uint(amin);   // (A) a0 >= a1; equal: a flat block, the 4+2 mode, every index 0
    float pa[8];
    palette(a0, a1, pa);
    uint idx[16];
    float err = alpha_fit(al, pa, idx);
    if (has_mid && has_end) {   // (B) the endpoints bracket the texels that are not 0 or 255
        uint b0 = uint(mid_min), b1 = uint(mid_max);
        float pb[8];
        palette(b0, b1, pb);
        uint idx2[16];
        float err2 = b0 < b1 ? alpha_fit(al, pb, idx2) : 1e9;   // b0 == b1 would be palette A's own flat mode
        if (err2 < err) {
            a0 = b0, a1 = b1, err = err2;
            for (uint i = 0u; i < 16u; i++) idx[i] = idx2[i];
        }
    }
    if (a0 == a1)
        for (uint i = 0u; i < 16u; i++) idx[i] = 0u;
    uvec2 r = uvec2(a0 | (a1 << 8), 0u);
    for (uint i = 0u; i < 16u; i++) {
        uint bit = 16u + 3u * i, w = bit >> 5, s = bit & 31u;
        r[w] |= idx[i] << s;
        if (s > 29u) r[w + 1u] |= idx[i] >> (32u - s);
    }
    return r;
}

#endif

void main() {
    uint bxi = gl_GlobalInvocationID.x, byi = gl_GlobalInvocationID.y, z = gl_GlobalInvocationID.z;
    if (bxi >= bx || byi * bx >= bslice) return;
    vec3 c[16];
    float al[16];
    for (uint i = 0u; i < 16u; i++) {
        vec4 t = texel(bxi * 4u + (i & 3u), byi * 4u + (i >> 2), z);
        c[i] = t.rgb, al[i] = t.a;
    }
    blk[out_base + z * bslice + byi * bx + bxi] = uvec4(encode_alpha(al), encode_colour(c));
}
