# Cleaned TACO final-checkpoint coding evaluation

All models: global_step_66; DP=8 / TP=1; greedy; max completion 32768; same benchmark prompts.
LCB v6 reuses identical v5 tasks plus 175 new tasks. Plus scores are enhanced-test pass@1.

| Model | Thinking | Dataset | Passed / total | Pass@1 | Mean completion tokens | Length limit |
|---|---|---|---:|---:|---:|---:|
| off_off | off | lcb_v5 | 397/880 | 0.4511 | 8108.2 | 193 |
| off_off | off | lcb_v6 | 452/1055 | 0.4284 | 9127.0 | 265 |
| off_off | off | lcb_v6_new175 | 55/175 | 0.3143 | 14250.2 | 72 |
| off_on | off | lcb_v5 | 410/880 | 0.4659 | 23515.6 | 546 |
| off_on | off | lcb_v6 | 462/1055 | 0.4379 | 23949.4 | 674 |
| off_on | off | lcb_v6_new175 | 52/175 | 0.2971 | 26131.1 | 128 |
| on_on | off | lcb_v5 | 407/880 | 0.4625 | 5752.0 | 120 |
| on_on | off | lcb_v6 | 466/1055 | 0.4417 | 6393.7 | 160 |
| on_on | off | lcb_v6_new175 | 59/175 | 0.3371 | 9620.4 | 40 |
