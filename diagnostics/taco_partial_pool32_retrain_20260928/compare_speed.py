import datetime
import json
from pathlib import Path
import re
import statistics

ROOT = Path('/home/liyu.468/learning_kvcache')
EXP = ROOT / 'outputs/taco_partial_pool32_retrain_20260928_1830'
OLD = ROOT / 'outputs/prefix_composition_3domain_20260922_4gpu/logs/taco_4gpu_full_20260923_022329.log'
pattern = r'([\w/.-]+):\s*(?:np\.\w+\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)'
old = []
for line in re.sub(r'\x1b\[[0-9;]*m', '', OLD.read_text()).splitlines():
    if re.search(r'\bstep:\s*\d+\s*-', line):
        row = {k:float(v) for k,v in re.findall(pattern, line)}
        if 'timing_s/step' in row and 'timing_s/gen' in row:
            old.append(row)
current = [json.loads(line) for line in (EXP / 'metrics.jsonl').read_text().splitlines()]
current = [r for r in current if 'timing_s/step' in r and 'timing_s/gen' in r]
last = max(r['step'] for r in current)
def select(rows, limit=None):
    return [r for r in rows if r['step'] > 1 and (limit is None or r['step'] <= limit)
            and not any(r.get(k, 0) > 0 for k in ['timing_s/testing', 'timing_s/validation', 'timing_s/validate'])]
def stats(rows):
    result={'count':len(rows), 'steps':[r['step'] for r in rows]}
    for k in ['timing_s/step','timing_s/gen','timing_s/update_actor','timing_s/old_log_prob','response_length/mean','response_length/clip_ratio','critic/score/mean']:
        vals=[r[k] for r in rows if k in r]
        if vals: result[k]=statistics.mean(vals)
    result['median_step_s']=statistics.median(r['timing_s/step'] for r in rows)
    result['samples_per_hour']=24*len(rows)*3600/sum(r['timing_s/step'] for r in rows)
    result['response_tokens_per_second']=24*sum(r['response_length/mean'] for r in rows)/sum(r['timing_s/step'] for r in rows)
    result['total_tokens_per_second']=sum(r['perf/total_num_tokens'] for r in rows)/sum(r['timing_s/step'] for r in rows)
    return result
new=stats(select(current))
full=stats(select(old))
matched=stats(select(old,last))
def compare(baseline):
    ratio=baseline['timing_s/step']/new['timing_s/step']
    return {'sample_throughput_ratio':ratio, 'sample_throughput_increase_percent':100*(ratio-1),
            'step_time_reduction_percent':100*(1-1/ratio),
            'response_token_throughput_ratio':new['response_tokens_per_second']/baseline['response_tokens_per_second']}
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'completed_training_steps':last,
        'current':new,'historical_all':full,'historical_same_steps':matched,
        'vs_historical_all':compare(full),'vs_historical_same_steps':compare(matched),
        'per_step_current':[{k:r[k] for k in ['step','timing_s/step','timing_s/gen','timing_s/update_actor','response_length/mean']} for r in current],
        'historical_log':str(OLD),
        'scope':'Same 3 actor + 1 teacher layout, batch 24 and response limit 16K. Historical hostname/cards not verified; outputs/sample order and normalization differ. Startup and validation are excluded.'}
(EXP/'speed_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
