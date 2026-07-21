---
configs:
  - config_name: bundled_shopping
    data_files:
      - split: test
        path: bundled_shopping/*.jsonl
  - config_name: progressive_search
    data_files:
      - split: test
        path: progressive_search/*.jsonl
  - config_name: group_travel_planner
    data_files:
      - split: test
        path: group_travel_planner/*.jsonl
  - config_name: formal_reasoning_math
    data_files:
      - split: test
        path: formal_reasoning_math/*.jsonl
  - config_name: formal_reasoning_phys
    data_files:
      - split: test
        path: formal_reasoning_phys/*.jsonl
---

# MemoryArena Dataset

## Overview
This dataset contains structured multi-session agentic tasks with question ```[list]```, answer ```[list]``` with necessary background context. Each row in the ```jsonl``` represents a agentic task ```[dict]``` with multiple subtasks, their corresponding answers, and background information.

## Dataset Structure

Each line in the JSONL file is a dictionary with the following fields:

- **id** (int): Unique identifier for each agentic task entry
- **questions** (list of str): List of sub-task queries
- **answers** (list of str): List of corresponding answers for each subtask
- **backgrounds** (str, or list of str): List of background/context information for each task
  - In Bundled Shopping and Progressive Search: no necessary backgrounds.
  - In Travel Planner: the travel details of the base person in each task serve as the background information for all subtasks.
  - In Formal Reasoning (Math and Phys): each subtask may has its background information. 

## Usage

Load with Hugging Face Datasets
```python
from datasets import load_dataset

ds = load_dataset("ZexueHe/memoryarena", "bundled_shopping")
ds = load_dataset("ZexueHe/memoryarena", "progressive_search")
ds = load_dataset("ZexueHe/memoryarena", "group_travel_planner")
ds = load_dataset("ZexueHe/memoryarena", "formal_reasoning_math")
ds = load_dataset("ZexueHe/memoryarena", "formal_reasoning_phys")
```

### Example task
In Bundled Webshop:
```json
{
  "id": 0,
  "questions": [
    "search subtask 1",
    "search subtask 1",
    ...
  ],
  "answers": [
    "search subtask result 1",
    "search subtask result 2",
    ...
  ]
}
```
In Progressive Search:
```json
{
  "id": 0,
  "questions": [
    "buy subtask item1",
    "buy subtask item 2",
    ...
  ],
  "answers": [
    {"target_asin": "B00TUDFEW2", "attributes": ["Almond Flour",...]},
    {"target_asin": "B08957C9ZH", "attributes": [...]},
    ...
  ]
}
```

In Group Travel Planner:

```json
{
  "id": 0,
  "base_person":{
    "name": "Jennifer",
    "query": "I am Jennifer. Please help me plan a trip from St. Petersburg to Rockford spanning 3 days...", # The travel requirements of the base person.
    "daily_plans": [ 
      {"days": 1, "current_city": "from St. Petersburg to Rockford", "transportation": "..."},
      {"days": 2, "current_city": "Rockford", "transportation": "..."},
      ...
    ], # Detailed plans of the base person.
  },
  "questions": [
    "I am Eric.\n I'm joining Jennifer for this trip...",
    "I am Emma.\n I'm traveling with Jennifer and Eric... ",
    ...
  ], # subtask intructions
  "answers": [
   [ 
      {"days": 1, "current_city": "from St. Petersburg to Rockford", "transportation": "..."},
      {"days": 2, "current_city": "Rockford", "transportation": "..."},
      ...
    ], # The answer of Subtask 1
    [ 
      {"days": 1, "current_city": "from St. Petersburg to Rockford", "transportation": "..."},
      {"days": 2, "current_city": "Rockford", "transportation": "..."},
      ...
    ], # The answer of Subtask 2
    ....
  ]
}
```

In Formal Reasoning (Math and Phys)
```json
{
  "id": 0,
  "paper_name": "paper_id", # which paper the questions are created from 
  "backgrounds": [
    "necessary definitions, formulations, and relevant context of subtask 1",
    "necessary definitions, formulations, and relevant context of subtask 2",
    ...
  ]
  "questions": [
    "Math subtask question 1",
    "Math subtask question 2",
    ...
  ],
  "answers": [
    "Math result for subtask 1",
     "Math result for subtask 2",
    ...
  ]
}
```



## License
This dataset is licensed under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## Citation
If you use this dataset, please cite:
```bibtex
@article{he2026memoryarena,
  title={MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks},
  author={He, Zexue and Wang, Yu and Zhi, Churan and Hu, Yuanzhe and Chen, Tzu-Ping and Yin, Lang and Chen, Ze and Wu, Tong Arthur and Ouyang, Siru and Wang, Zihan and others},
  journal={arXiv preprint arXiv:2602.16313},
  year={2026}
}
```
