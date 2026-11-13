"""DM_VK_SPV_KILLS (gfx_vk_shader.c vk_spv_kills): for a fragment shader, the instructions each discard's condition is computed
from. A hand-built module discards when its input's x is under 0.5; the lines name the input, the constant and the comparison."""
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
#define VKLOG(...) (printf(__VA_ARGS__), puts(""))
@FN@
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    static uint32_t w[4096]; uint32_t nw = (uint32_t)fread(w, 4, 4096, f); fclose(f);
    vk_spv_kills(w, nw, 0xabcd); return 0;
}
'''


def ins(op, *ops):
    return [(len(ops) + 1) << 16 | op, *ops]


def module(discard):
    """if (in0.x < 0.5) discard (or return); out0 = in0."""
    VOID, FNT, F32, V4, PIN4, POUT4, IN, OUT, MAIN, L0, V, U32, C0, PINF, BOOL, CHALF, A, X, CND, LT, LF = range(1, 22)
    w = ins(17, 1) + ins(14, 0, 1) + ins(15, 4, MAIN, 0x6e69616d, 0, IN, OUT) + ins(16, MAIN, 7) + ins(71, IN, 30, 0) + ins(71, OUT, 30, 0)
    w += ins(19, VOID) + ins(33, FNT, VOID) + ins(22, F32, 32) + ins(23, V4, F32, 4) + ins(32, PIN4, 1, V4) + ins(59, PIN4, IN, 1) + ins(32, POUT4, 3, V4) + ins(59, POUT4, OUT, 3)
    w += ins(21, U32, 32, 0) + ins(43, U32, C0, 0) + ins(32, PINF, 1, F32) + ins(20, BOOL) + ins(43, F32, CHALF, struct.unpack('<I', struct.pack('<f', 0.5))[0])
    w += ins(54, VOID, MAIN, 0, FNT) + ins(248, L0) + ins(65, PINF, A, IN, C0) + ins(61, F32, X, A) + ins(184, BOOL, CND, X, CHALF) + ins(247, LF, 0) + ins(250, CND, LT, LF)
    w += ins(248, LT) + ins(252 if discard else 253)
    w += ins(248, LF) + ins(61, V4, V, IN) + ins(62, OUT, V) + ins(253) + ins(56)
    return [0x07230203, 0x00010000, 0, 22, 0] + w


@unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
class SpvKills(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = (ROOT / 'nxnative/dm/android/gfx_vk_shader.c').read_text(encoding='utf-8')
        a = src.index('static const char *spv_opname('); b = src.index('\n}\n', src.index('void vk_spv_kills(')) + 3
        cls.d = Path(tempfile.mkdtemp(prefix='nx2a_spv_kills_')); c = cls.d / 't.c'
        c.write_text(HARNESS.replace('@FN@', src[a:b]), encoding='utf-8')
        cls.exe = cls.d / ('t.exe' if shutil.which('cc').lower().endswith('.exe') else 't')
        subprocess.run(['cc', '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-o', str(cls.exe), str(c)], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def lines(self, w):
        (self.d / 'in.spv').write_bytes(struct.pack(f'<{len(w)}I', *w))
        return subprocess.run([str(self.exe), str(self.d / 'in.spv')], capture_output=True, text=True, check=True).stdout.splitlines()

    def test_a_discard_lists_what_its_condition_reads(self):
        out = self.lines(module(True))
        self.assertEqual(out[0], 'spv 000000000000abcd: discard 1 when %19 is true')
        body = [l.split(' = ', 1)[1] for l in out[1:]]
        self.assertEqual(body, ['variable (input) location 0', 'constant 0', 'constant 0.5', 'AccessChain %7 %13', 'Load %17', 'FOrdLessThan %18 %16'])

    def test_a_shader_that_only_returns_has_no_discard(self):
        out = self.lines(module(False))
        self.assertEqual(len(out), 1); self.assertIn('no discard', out[0])


if __name__ == '__main__':
    unittest.main()
