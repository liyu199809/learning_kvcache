# Cleaned TACO final-checkpoint coding evaluation

All models: global_step_66; DP=8 / TP=1; greedy; max completion 16384; same benchmark prompts.
LCB v6 reuses identical v5 tasks plus 175 new tasks. Plus scores are enhanced-test pass@1.

| Model | Thinking | Dataset | Passed / total | Pass@1 | Mean completion tokens | Length limit |
|---|---|---|---:|---:|---:|---:|
| off_off | off | humaneval_plus | 124/164 | 0.7561 | 650.6 | 4 |
| off_off | off | mbpp_plus | 248/378 | 0.6561 | 333.7 | 2 |
| off_off | off | lcb_v5 | 403/880 | 0.4580 | 4357.7 | 182 |
| off_off | off | lcb_v6 | 464/1055 | 0.4398 | 4950.4 | 258 |
| off_off | off | lcb_v6_new175 | 60/175 | 0.3429 | 7930.5 | 76 |
| off_on | off | humaneval_plus | 135/164 | 0.8232 | 3032.6 | 20 |
| off_on | off | mbpp_plus | 258/378 | 0.6825 | 1550.9 | 21 |
| off_on | off | lcb_v5 | 399/880 | 0.4534 | 13147.7 | 567 |
| off_on | off | lcb_v6 | 454/1055 | 0.4303 | 13270.6 | 695 |
| off_on | off | lcb_v6_new175 | 55/175 | 0.3143 | 13888.5 | 128 |
| on_on | off | humaneval_plus | 130/164 | 0.7927 | 371.9 | 1 |
| on_on | off | mbpp_plus | 249/378 | 0.6587 | 476.7 | 5 |
| on_on | off | lcb_v5 | 412/880 | 0.4682 | 3506.6 | 120 |
| on_on | off | lcb_v6 | 474/1055 | 0.4493 | 4012.3 | 173 |
| on_on | off | lcb_v6_new175 | 62/175 | 0.3543 | 6555.1 | 53 |
