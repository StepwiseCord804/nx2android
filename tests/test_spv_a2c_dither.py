"""Alpha to coverage on a single-sample target as a dither (gfx_vk_shader.c spv_a2c_dither): the fragment shader's variant
discards a fragment whose alpha at output 0 is under a 4x4 ordered-dither threshold. A small SPIR-V interpreter runs the
variant: alpha k/16 keeps k pixels of each 4x4 block, spread as a Bayer matrix, at every return of the entry point."""
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@SPVI@
@FN@
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    static uint32_t w[4096]; uint32_t nw = (uint32_t)fread(w, 4, 4096, f), n2 = 0; fclose(f);
    uint32_t *o = spv_a2c_dither(w, nw, &n2); if (!o) { puts("NULL"); return 0; }
    f = fopen(argv[2], "wb"); fwrite(o, 4, n2, f); fclose(f); free(o); puts("OK"); return 0;
}
'''


def ins(op, *ops):
    return [(len(ops) + 1) << 16 | op, *ops]


def fl(v):
    return struct.unpack('<I', struct.pack('<f', v))[0]


def module(two_returns, out_type='vec4', index1=False):
    """out0 = in0 (location 0 each), in one block or with an early return when in0.x < 0.5."""
    VOID, FNT, F32, V4, PIN4, POUT4, IN, OUT, MAIN, L0, V, U32, C0, PINF, BOOL, CHALF, A, X, CND, LT, LF, V2, I32, IV4, POUTI = range(1, 26)
    w = ins(17, 1) + ins(14, 0, 1) + ins(15, 4, MAIN, 0x6e69616d, 0, IN, OUT) + ins(16, MAIN, 7) + ins(71, IN, 30, 0) + ins(71, OUT, 30, 0)
    if index1: w += ins(71, OUT, 32, 1)
    w += ins(19, VOID) + ins(33, FNT, VOID) + ins(22, F32, 32) + ins(23, V4, F32, 4) + ins(32, PIN4, 1, V4) + ins(59, PIN4, IN, 1)
    if out_type == 'vec4': w += ins(32, POUT4, 3, V4) + ins(59, POUT4, OUT, 3)
    else: w += ins(21, I32, 32, 1) + ins(23, IV4, I32, 4) + ins(32, POUTI, 3, IV4) + ins(59, POUTI, OUT, 3)
    if two_returns: w += ins(21, U32, 32, 0) + ins(43, U32, C0, 0) + ins(32, PINF, 1, F32) + ins(20, BOOL) + ins(43, F32, CHALF, fl(0.5))
    w += ins(54, VOID, MAIN, 0, FNT) + ins(248, L0)
    if out_type != 'vec4': w += ins(253) + ins(56)
    elif two_returns:
        w += ins(65, PINF, A, IN, C0) + ins(61, F32, X, A) + ins(184, BOOL, CND, X, CHALF) + ins(247, LF, 0) + ins(250, CND, LT, LF)
        w += ins(248, LT) + ins(61, V4, V, IN) + ins(62, OUT, V) + ins(253)
        w += ins(248, LF) + ins(61, V4, V2, IN) + ins(62, OUT, V2) + ins(253) + ins(56)
    else: w += ins(61, V4, V, IN) + ins(62, OUT, V) + ins(253) + ins(56)
    return [0x07230203, 0x00010000, 0, 26, 0] + w


def run(w, frag, in0):
    """Interpret a fragment module: True when the fragment is kept, False when it is discarded."""
    deco, var, const, types, mem, val = {}, {}, {}, {}, {}, {}
    i, code = 5, []
    while i < len(w):
        n, op = w[i] >> 16, w[i] & 0xffff; a = w[i + 1:i + n]
        if op == 71 and a[1] == 11: deco[a[0]] = a[2]
        elif op in (20, 21, 22, 23, 32): types[a[0]] = (op, a[1:])
        elif op == 43: const[a[1]] = struct.unpack('<f', struct.pack('<I', a[2]))[0] if types[a[0]][0] == 22 else a[2]
        elif op == 59: var[a[1]] = a[2]
        elif op in (54, 56): pass
        elif op not in (17, 14, 15, 16, 71, 19, 33): code.append((op, a))
        i += n
    for v, cls in var.items(): mem[v] = list(frag) if deco.get(v) == 15 else list(in0) if cls == 1 else [None] * 4
    labels = {a[0]: k for k, (op, a) in enumerate(code) if op == 248}
    get = lambda x: const[x] if x in const else val[x]
    pc = 0
    while True:
        op, a = code[pc]; pc += 1
        if op == 248 or op == 247: continue
        if op == 253: return True
        if op == 252: return False
        if op == 250: pc = labels[a[1] if get(a[0]) else a[2]]; continue
        if op == 65: val[a[1]] = (a[2], get(a[3]))
        elif op == 61: p = get(a[2]) if a[2] in val else None; val[a[1]] = mem[p[0]][p[1]] if p else list(mem[a[2]])
        elif op == 62: mem[a[0]] = list(get(a[1]))
        elif op == 109: val[a[1]] = int(get(a[2]))
        elif op == 112: val[a[1]] = float(get(a[2]))
        elif op == 198: val[a[1]] = get(a[2]) ^ get(a[3])
        elif op == 199: val[a[1]] = get(a[2]) & get(a[3])
        elif op == 197: val[a[1]] = get(a[2]) | get(a[3])
        elif op == 196: val[a[1]] = get(a[2]) << get(a[3]) & 0xffffffff
        elif op == 194: val[a[1]] = get(a[2]) >> get(a[3])
        elif op == 133: val[a[1]] = get(a[2]) * get(a[3])
        elif op == 129: val[a[1]] = get(a[2]) + get(a[3])
        elif op == 184: val[a[1]] = get(a[2]) < get(a[3])
        else: raise AssertionError(f'op {op} is not interpreted')


@unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
class A2cDither(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = (ROOT / 'nxnative/dm/android/gfx_vk_shader.c').read_text(encoding='utf-8')
        a = src.index('typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt;'); b = src.index('static uint32_t spvi_vec4(')
        f = src.index('static uint32_t *spv_a2c_dither('); g = src.index('\n}\n', f) + 3
        cls.d = Path(tempfile.mkdtemp(prefix='nx2a_a2c_dither_')); c = cls.d / 't.c'
        c.write_text(HARNESS.replace('@SPVI@', src[a:b]).replace('@FN@', src[f:g]), encoding='utf-8')
        cls.exe = cls.d / ('t.exe' if shutil.which('cc').lower().endswith('.exe') else 't')
        subprocess.run(['cc', '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-o', str(cls.exe), str(c)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def variant(self, w):
        (self.d / 'in.spv').write_bytes(struct.pack(f'<{len(w)}I', *w))
        r = subprocess.run([str(self.exe), str(self.d / 'in.spv'), str(self.d / 'out.spv')], capture_output=True, text=True, check=True)
        if r.stdout.strip() == 'NULL': return None
        b = (self.d / 'out.spv').read_bytes(); o = list(struct.unpack(f'<{len(b) // 4}I', b))
        val = shutil.which('spirv-val')
        if val: subprocess.run([val, str(self.d / 'out.spv')], check=True)
        return o

    def test_alpha_keeps_that_share_of_each_block(self):
        for two in (False, True):
            o = self.variant(module(two)); self.assertIsNotNone(o)
            for x in (0.25, 0.75):   # each side of the early return
                for k in range(17):
                    kept = [[run(o, (bx + 0.5, by + 0.5, 0.5, 1.0), (x, 0.0, 0.0, k / 16)) for bx in range(8)] for by in range(8)]
                    for oy in (0, 4):
                        for ox in (0, 4): self.assertEqual(sum(kept[oy + j][ox + i] for j in range(4) for i in range(4)), k, (two, x, k))
                    if k == 8:   # half coverage is a checkerboard, not rows or blocks
                        for j in range(8):
                            for i in range(7): self.assertNotEqual(kept[j][i], kept[j][i + 1])
                    if k == 4:   # a quarter: one pixel of each 2x2
                        for j in range(0, 8, 2):
                            for i in range(0, 8, 2): self.assertEqual(kept[j][i] + kept[j][i + 1] + kept[j + 1][i] + kept[j + 1][i + 1], 1)

    def test_a_fragment_kept_at_an_alpha_stays_at_a_higher_one(self):
        o = self.variant(module(False))
        for by in range(4):
            for bx in range(4):
                seen = [run(o, (bx + 0.5, by + 0.5, 0.5, 1.0), (0.0, 0.0, 0.0, k / 16)) for k in range(17)]
                self.assertEqual(seen, sorted(seen))

    def test_no_variant_without_a_float_colour_at_location_0(self):
        self.assertIsNone(self.variant(module(False, out_type='ivec4')))
        self.assertIsNone(self.variant(module(False, index1=True)))


if __name__ == '__main__':
    unittest.main()
