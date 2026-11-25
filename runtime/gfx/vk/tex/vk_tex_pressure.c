#include <stdlib.h>
#include <string.h>
#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_tex_internal.h"

// ---------------------------------------------------------------- eviction under VRAM pressure (discrete GPUs)
// evict_idle and vk_texture_trim free textures only when an allocation fails. On Windows one rarely does: past the
// budget WDDM moves memory out of the GPU instead, and the game slows down. Issue #747 (RTX 3060 Ti 8 GB, 70 800
// frames of Grand Prix): VMA's blocks grew 4 456 -> 6 760 MiB (never given back) while the budget the system gave fell
// from 7 256 to ~6 650 MiB (other programs), and the mean frame rate fell 60 -> 35-42 fps once the driver's usage
// reached ~7 090 MiB; most of the growth was decoded ASTC textures (491 -> 1 798 MiB), mirrors of CPU-written
// storage that come back from it on demand. Same report in #481 #478 #723 #757 #783 #730.
// Here, every PRESSURE_EVERY submits: with VMA's device-local blocks past PRESSURE_HIGH % of the live budget, the
// recreatable textures (evict_idle's rule) idle the longest go, oldest use first, until VMA's allocations are back at
// PRESSURE_LOW %. New images then fill the holes instead of making blocks, and a block left empty is given back.
// Blocks past the line with allocations already under it are fragmentation, which freeing more does not fix.
// The other way past the budget: at MKW_GPU_MEM_LIMIT's cap on heap 0, AUTO_PREFER_DEVICE puts new images in the host
// heap, and NVIDIA's driver keeps them in VRAM all the same. On 4 GB cards (#56, #126) the driver's heap 0 usage passed
// VMA's heap 0 blocks by what VMA held in heap 1 (957/960, 1 497/1 474, 2 198/2 190 MiB) up to 5.2 GB, and the game
// fell to a few fps after minutes. With the driver's usage over the budget, the idle textures outside device-local
// memory go first, all of them: a heap 1 block left empty is given back, and the driver's count with it.
// The idle line is long on purpose: dropping the released textures at the budget (windows win5, 06/11/2026) took
// GPU frames from 17-19 to 47-50 ms on a GTX 1050 Ti, as the game asked for them again within frames and each one
// was made and uploaded anew. MKW_TEX_PRESSURE=<submits> sets that line (default 1800, ~10 s at ~3 submits a frame),
// 0 turns it off, "ab" switches it with the A/B blocks.
// Off unless set: on a GTX 1050 Ti (windows prsC, 23/11/2026, 25 min of races) the textures outside VRAM were never
// idle enough to go (all 7.9 GB it evicted were in VRAM; they came and went with the game, 31-476 MiB), and the frame rate
// held at 23-26 fps from start to end; the runs without it (prsA, prsB) slowed only on that PC's disk (pipeline
// cache writes of 29-370 s).
#define PRESSURE_EVERY 8
#define PRESSURE_HIGH 90
#define PRESSURE_LOW 80
#define PRESSURE_RETRY 600   // submits before scanning again after a pass that could not reach PRESSURE_LOW
#define PRESSURE_SETTLE 64   // after one that did: the images go once the parts sent are done, and VMA counts them
                             // until then (a pass 8 submits later would evict again for the same bytes)

static uint32_t idle_line(int *mode) {
    static int m = -1;
    static uint32_t line;
    if (m < 0) {
        const char *e = cfg_get("MKW_TEX_PRESSURE");
        m = rt_knob("TEX_PRESSURE", "0");   // off by default: no gain measured yet (header)
        long v = e && *e && m == 1 ? strtol(e, NULL, 10) : 0;
        line = v > 1 ? (uint32_t)v : 1800;
        rt_log("vk: VRAM pressure eviction %s (MKW_TEX_PRESSURE: idle %u submits, past %d%% of the budget down to %d%%)",
               !m ? "off" : m == 2 ? "in the B blocks" : "on", line, PRESSURE_HIGH, PRESSURE_LOW);
    }
    *mode = m;
    return line;
}

static uint32_t age_of(const Tex *e) { return submit_epoch - e->seen; }

static int oldest_first(const void *a, const void *b) {
    uint32_t x = age_of(*(Tex *const *)a), y = age_of(*(Tex *const *)b);
    return (x < y) - (x > y);
}

typedef struct { uint64_t blocks, allocs, budget, usage; } Heaps;

static Heaps device_local(VkCtx *c) {
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);   // the live budget (VK_EXT_memory_budget), under MKW_GPU_MEM_LIMIT's heap cap
    Heaps s = {0};
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            s.blocks += vb[h].statistics.blockBytes, s.allocs += vb[h].statistics.allocationBytes,
                s.budget += vb[h].budget, s.usage += vb[h].usage;   // usage: the driver's count, all of the process
    return s;
}

static int outside_vram(VkCtx *c, uint32_t type) {
    return type < c->mem_props.memoryTypeCount &&
           !(c->mem_props.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// The texture lock held (vk_texture_new_submit), on the replay thread: a check of VMA's counters every
// PRESSURE_EVERY submits; the walk over the textures only past the line.
void tex_pressure(void) {
    static uint32_t next_at;
    static uint64_t passes, n_out, bytes_out;
    int mode;
    uint32_t line = idle_line(&mode);
    if (!RT_KNOB_ON(mode) || submit_epoch % PRESSURE_EVERY || (int32_t)(submit_epoch - next_at) < 0) return;
    VkCtx *c = vk_ctx();
    if (!c || !c->vma) return;
    Heaps s = device_local(c);
    uint64_t low = s.budget / 100 * PRESSURE_LOW;
    int packed = s.blocks > s.budget / 100 * PRESSURE_HIGH && s.allocs > low, spilled = s.usage > s.budget;
    if (!s.budget || (!packed && !spilled)) return;
    Tex **cand = malloc(sizeof *cand * (size_t)(by_key.n + 1));
    if (!cand) return;
    int n = 0;
    for (uint32_t i = 0; i < by_key.cap; i++) {
        Tex *e = by_key.vals[i];
        if (e && e->alloc && e->pub.mirrored && !e->pub.gpu_owned && !e->gpu_dirty && age_of(e) > line)
            cand[n++] = e;
    }
    qsort(cand, (size_t)n, sizeof *cand, oldest_first);
    uint64_t need = packed ? s.allocs - low : 0, freed = 0, host = 0;
    int out = 0;
    for (int i = 0; i < n && (spilled || freed < need); i++) {   // past the budget: every idle one outside VRAM too
        VmaAllocationInfo vi = {0};
        vmaGetAllocationInfo(c->vma, cand[i]->alloc, &vi);
        int away = outside_vram(c, vi.memoryType);
        if (away ? !spilled : freed >= need) continue;
        if (away) host += vi.size;
        else freed += vi.size;
        destroy_idle(cand[i]), out++;
    }
    free(cand);
    next_at = submit_epoch + (freed < need || (spilled && !host) ? PRESSURE_RETRY : PRESSURE_SETTLE);
    if (!out) return;   // the driver's usage over the budget with nothing outside VRAM: its own memory, not ours
    passes++, n_out += (uint64_t)out, bytes_out += freed + host;
    if (passes <= 8 || !(passes & (passes - 1)))
        rt_log("vk: VRAM pressure: blocks %llu of %llu MiB budget, allocations %llu MiB, driver usage %llu MiB: %d idle "
               "textures out (%llu MiB in VRAM, %llu MiB outside it; %d were idle enough); %llu passes, %llu textures, "
               "%llu MiB in all",
               (unsigned long long)(s.blocks >> 20), (unsigned long long)(s.budget >> 20),
               (unsigned long long)(s.allocs >> 20), (unsigned long long)(s.usage >> 20), out,
               (unsigned long long)(freed >> 20), (unsigned long long)(host >> 20), n, (unsigned long long)passes,
               (unsigned long long)n_out, (unsigned long long)(bytes_out >> 20));
}
