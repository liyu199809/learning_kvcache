"""Compare full new-model outputs with cached historical benchmark scores."""
import csv
import json
from pathlib import Path

ROOT = Path('/home/liyu.468/learning_kvcache')
OUT = ROOT/'outputs/b300_taco_compare_20260929'
HIST = ROOT/'outputs/prefix_composition_3domain_20260922_4gpu/eval/serial_dp4_20260923_161445/taco'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def score(path, suite):
    if suite in ('humaneval+', 'mbpp+', 'livecodebench'):
        summary = read(path/'summary.json')
        if not summary:
            return None
        metrics = summary['metrics']
        return dict(score=metrics['pass@1'] if suite == 'livecodebench' else metrics['plus']['pass@1'],
                    tasks=summary['tasks'], generation_errors=summary['generation_errors'],
                    metric='pass@1', timing=read(path/'timing.json'))
    if suite == 'bfcl_v3':
        p = path/'score/data_overall.csv'
        if not p.exists():
            return None
        row = list(csv.DictReader(p.open()))[0]
        return dict(score=float(row['Overall Acc'].strip('%'))/100,
                    metric='Overall Acc', timing=read(path/'timing.json'))
    result = read(path/'results.json')
    if not result:
        return None
    simulations = result['simulations']
    expected = 50 if suite == 'tau2_airline' else 114
    trials = result.get('info', {}).get('num_trials', 1)
    if len(result['tasks']) != expected or len(simulations) != expected * trials:
        return None
    # tau2's official total reward is the joint task-success criterion.
    rewards = [(item.get('reward_info') or {}).get('reward', 0) for item in simulations]
    return dict(score=sum(rewards)/len(rewards), tasks=len(result['tasks']),
                simulations=len(simulations), metric='mean reward / pass@1',
                timing=read(path/'timing.json'))


def main():
    result = {'comparison':{}, 'historical_old':{}, 'status':{},
              'note':'New model only on B300 SGLang. Old scores are reused from the existing full vLLM run. No matched inference-speed comparison.'}
    lines = ['# B300 TACO prefix 完整评测对比', '',
             '只运行新模型，旧分数复用已有完整评测；未加载、重跑旧模型。',
             '新模型：partial rollout pool32/train24，step66；旧模型：同步训练，step66。',
             '使用 B300 现有 SGLang 0.5.15.post1，TP1/DP4；未安装新的推理环境。',
             '采样：temperature=0.6、top_p=1；代码 seed42、thinking on、16K tokens、并发64。tau2 Airline 并发2、Retail 并发16；单次 trial、最多200步，agent thinking off。旧 tau2 每域并发2。', '',
             '| Benchmark | 旧模型已有结果 | 新模型 SGLang | 变化（百分点） |',
             '|---|---:|---:|---:|']
    for suite in ('humaneval+', 'mbpp+', 'livecodebench', 'bfcl_v3', 'tau2_airline', 'tau2_retail'):
        old, new = score(HIST/suite,suite), score(OUT/'new'/suite,suite)
        row = dict(old=old, new=new)
        if old and new:
            row['delta_percentage_points'] = (new['score']-old['score'])*100
        result['comparison'][suite] = row
        result['historical_old'][suite] = old
        values = [f"{item['score']*100:.2f}%" if item else 'pending' for item in (old,new)]
        delta = f"{row['delta_percentage_points']:+.2f}" if 'delta_percentage_points' in row else 'pending'
        lines.append(f'| {suite} | {values[0]} | {values[1]} | {delta} |')
    for name in ('new',):
        result['status'][name] = read(OUT/name/'status.json')
    lines += ['', '## 推理耗时', '',
              '下表只统计新模型代码生成阶段；不含模型加载和代码判分。', '',
              '| Benchmark | 任务数 | 生成耗时（分钟） | 输出 tokens/s |',
              '|---|---:|---:|---:|']
    for suite in ('humaneval+', 'mbpp+', 'livecodebench'):
        new = result['comparison'][suite]['new']
        timing = new and new.get('timing')
        if timing:
            lines.append(f"| {suite} | {new['tasks']} | {timing['generation_seconds']/60:.2f} | {timing['completion_tokens_per_second']:,.0f} |")
    lines += ['', '## 对比边界与复现', '',
              '旧评测使用 H200/vLLM，新评测使用 B300/SGLang；引擎、硬件和代码推理并发不同。未做相同推理配置的速度对照，不能由此给出推理加速倍率。',
              '新训练同时修正了纯蒸馏梯度归一化，因此分数变化不能全部归因于 partial rollout。单次采样分数也存在随机波动。',
              '三个代码集的任务、版本与主要采样参数已核对，见 protocol_audit.json；tau2 完整核对见 tau2_protocol_audit.json。',
              '自定义 hybrid prefix 在 SGLang 原生模型上的适配严格加载112个 prefix tensors；两条提示的首 token 与续写 greedy tokens 对齐 HF，验证产物为 hf_parity.json。',
              'tau2 使用 B300 /usr/bin/python3 加载已有 tau2 site-packages；任务目录中的 entrypoint 仅解决原 H200 Python 路径问题，没有安装依赖或改动共享环境。',
              f'新模型：`{ROOT}/outputs/taco_partial_pool32_retrain_20260928_1830/prefixes/taco`',
              f'旧结果：`{HIST}`',
              f'新结果：`{OUT}/new`']
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
