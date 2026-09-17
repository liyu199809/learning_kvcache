"""Concatenate ordered, independently trained hybrid-prefix HF models on CPU."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from prefix_tuning.virtual_prefix.merge_hybrid_prefix_partitions import (
    DELTA_FILE, ATTENTION_FILE, DELTA_SUFFIXES, ATTENTION_SUFFIXES, _copy_source,
)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def base_config(config):
    c = copy.deepcopy(config)
    for key in ('independent_delta_kv_prefix', 'residual_attention_prefix',
                'hybrid_prefix', '_name_or_path'):
        c.pop(key, None)
    for key in list(c.get('text_config', {})):
        if key.startswith(('independent_delta_prefix_', 'residual_attention_prefix_')):
            c['text_config'].pop(key)
    return c


def compose(inputs: list[Path], output: Path):
    inputs = [p.resolve() for p in inputs]
    output = output.resolve()
    if not inputs:
        raise ValueError('At least one prefix is required')
    if output.exists():
        raise FileExistsError(output)
    configs = [json.loads((p / 'config.json').read_text()) for p in inputs]
    indexes = [json.loads((p / 'model.safetensors.index.json').read_text()) for p in inputs]
    if any(not c.get('hybrid_prefix') for c in configs):
        raise ValueError('Only hybrid prefix HF checkpoints are supported')
    if any(base_config(c) != base_config(configs[0]) for c in configs[1:]):
        raise ValueError('Base model configurations differ')
    maps = [{n: f for n, f in idx['weight_map'].items()
             if not n.endswith(DELTA_SUFFIXES + ATTENTION_SUFFIXES)} for idx in indexes]
    if any(m != maps[0] for m in maps[1:]):
        raise ValueError('Base weight indexes differ')
    hashes = {}
    for name in sorted(set(maps[0].values())):
        ref = inputs[0] / name
        hashes[name] = digest(ref)
        for p in inputs[1:]:
            if not os.path.samefile(ref, p / name) and digest(p / name) != hashes[name]:
                raise ValueError(f'Base weight mismatch: {p / name}')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        ref = inputs[0] / name
        for p in inputs[1:]:
            other = p / name
            if ref.exists() != other.exists() or (ref.exists() and digest(ref) != digest(other)):
                raise ValueError(f'Tokenizer/template mismatch: {name}')
    merged = {}
    lengths = {}
    for filename, section, suffixes in (
        (DELTA_FILE, 'independent_delta_kv_prefix', DELTA_SUFFIXES),
        (ATTENTION_FILE, 'residual_attention_prefix', ATTENTION_SUFFIXES),
    ):
        parts = [load_file(str(p / filename), device='cpu') for p in inputs]
        names = set(parts[0])
        if any(set(part) != names for part in parts) or not all(n.endswith(suffixes) for n in names):
            raise ValueError(f'Prefix names mismatch: {filename}')
        for name in names:
            tensors = [part[name] for part in parts]
            for tensor, c in zip(tensors, configs):
                if tensor.ndim != 2 or tensor.shape[0] != c[section]['num_virtual_tokens']:
                    raise ValueError(f'Invalid prefix shape: {name}')
                if tensor.shape[1:] != tensors[0].shape[1:] or tensor.dtype != tensors[0].dtype:
                    raise ValueError(f'Prefix dimension/dtype mismatch: {name}')
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Non-finite prefix: {name}')
            merged[name] = torch.cat(tensors, dim=0).contiguous()
        lengths[section] = sum(c[section]['num_virtual_tokens'] for c in configs)
    config = copy.deepcopy(configs[0])
    for section, length in lengths.items():
        suffixes = DELTA_SUFFIXES if section == 'independent_delta_kv_prefix' else ATTENTION_SUFFIXES
        selected = {n: t for n, t in merged.items() if n.endswith(suffixes)}
        config[section].update(num_virtual_tokens=length,
                               trainable_tensor_count=len(selected),
                               trainable_numel=sum(t.numel() for t in selected.values()))
    g, a = lengths['independent_delta_kv_prefix'], lengths['residual_attention_prefix']
    config['text_config']['independent_delta_prefix_num_virtual_tokens'] = g
    config['text_config']['residual_attention_prefix_num_virtual_tokens'] = a
    config['residual_attention_prefix']['prefix_positions'] = [-a, -1]
    config['hybrid_prefix'] = dict(delta_prefix_num_virtual_tokens=g,
                                   attention_prefix_num_virtual_tokens=a,
                                   trainable_tensor_count=len(merged),
                                   trainable_numel=sum(t.numel() for t in merged.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        _copy_source(inputs[0], staging, 'hardlink')
        for filename, suffixes in ((DELTA_FILE, DELTA_SUFFIXES), (ATTENTION_FILE, ATTENTION_SUFFIXES)):
            save_file({n: t for n, t in merged.items() if n.endswith(suffixes)}, str(staging / filename))
        idx = copy.deepcopy(indexes[0])
        old_bytes = sum(t.numel() * t.element_size() for f in (DELTA_FILE, ATTENTION_FILE)
                        for t in load_file(str(inputs[0] / f)).values())
        new_bytes = sum(t.numel() * t.element_size() for t in merged.values())
        idx['metadata']['total_size'] += new_bytes - old_bytes
        manifest = {'method': 'ordered_token_concatenation', 'ordered_inputs': list(map(str, inputs)),
                    'base_sha256': hashes, 'prefix_sha256': [
                        {f: digest(p / f) for f in (DELTA_FILE, ATTENTION_FILE)} for p in inputs],
                    **config['hybrid_prefix']}
        for name, value in [('config.json', config), ('model.safetensors.index.json', idx),
                            ('independent_delta_kv_prefix_config.json', config['independent_delta_kv_prefix']),
                            ('residual_attention_prefix_config.json', config['residual_attention_prefix']),
                            ('prefix_composition_manifest.json', manifest)]:
            (staging / name).write_text(json.dumps(value, indent=2) + '\n')
        for filename in (DELTA_FILE, ATTENTION_FILE):
            for name, tensor in load_file(str(staging / filename)).items():
                if not torch.equal(tensor, merged[name]):
                    raise AssertionError(f'Export changed tensor: {name}')
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.input, args.output), indent=2))
