import json
import shutil

import torch
from safetensors.torch import load_file, save_file

from prefix_tuning.tests.test_split_merge_hybrid_prefix import SplitMergeHybridPrefixTest
from prefix_tuning.virtual_prefix.compose_hybrid_prefixes import compose


class ComposeHybridPrefixTest(SplitMergeHybridPrefixTest):
    def test_ordered_independent_composition(self):
        other = self.root / 'independent'
        shutil.copytree(self.source, other)
        filename = 'residual_attention_prefix.safetensors'
        values = load_file(str(other / filename))
        save_file({n: t + 7 for n, t in values.items()}, str(other / filename))
        a, b = self.root / 'ab', self.root / 'ba'
        compose([self.source, other], a)
        compose([other, self.source], b)
        config = json.loads((a / 'config.json').read_text())
        self.assertEqual(config['text_config']['independent_delta_prefix_num_virtual_tokens'], 16)
        self.assertEqual(config['residual_attention_prefix']['prefix_positions'], [-8, -1])
        for n, t in load_file(str(a / filename)).items():
            torch.testing.assert_close(t[:4], values[n])
            torch.testing.assert_close(t[4:], values[n] + 7)
            reversed_t = load_file(str(b / filename))[n]
            torch.testing.assert_close(reversed_t[:4], t[4:])
        with self.assertRaises(FileExistsError):
            compose([self.source], a)

    def test_reject_changed_base(self):
        other = self.root / 'wrong_base'
        shutil.copytree(self.source, other)
        save_file({'base.weight': torch.zeros(2, 2, dtype=torch.bfloat16)}, str(other / 'base.safetensors'))
        with self.assertRaisesRegex(ValueError, 'Base weight mismatch'):
            compose([self.source, other], self.root / 'must_not_exist')
        self.assertFalse((self.root / 'must_not_exist').exists())

    def test_reject_prefix_dtype_mismatch(self):
        other = self.root / 'wrong_dtype'
        shutil.copytree(self.source, other)
        filename = 'independent_delta_kv_prefix.safetensors'
        save_file({n: t.float() for n, t in load_file(str(other / filename)).items()}, str(other / filename))
        with self.assertRaisesRegex(ValueError, 'dtype mismatch'):
            compose([self.source, other], self.root / 'must_not_exist')
