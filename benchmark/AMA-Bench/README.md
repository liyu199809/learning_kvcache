---
license: mit
task_categories:
- question-answering
---

# AMA-Bench: Open-Ended QA for Long-Horizon Memory Evaluation

This repository contains the Open-Ended QA dataset for **AMA-Bench**, introduced in the paper [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](https://arxiv.org/pdf/2602.22769).

## 🔗 Quick Links

- **[Website](https://ama-bench.github.io/)** - Official AMA-Bench website
- **[GitHub Repository](https://github.com/AMA-Bench/AMA-Hub)** - Complete AMA-Bench project
- **[Dataset](https://huggingface.co/datasets/AMA-bench/AMA-bench)** - Full dataset on Hugging Face
- **[Leaderboard](https://huggingface.co/spaces/AMA-bench/AMA-bench-Leaderboard)** - Model rankings and results
- **[Paper](https://arxiv.org/pdf/2602.22769)** - Research paper

## Overview

AMA-Bench Open-Ended QA is designed to evaluate the long-horizon memory capabilities of AI agents across various agentic applications. The dataset contains agent trajectories paired with open-ended questions to assess an agent's ability to retain and recall information throughout extended task execution.

### Key Features

- **Comprehensive evaluation**: Tests multiple aspects of memory (recall, causal inference, state tracking, and abstraction)
- **Diverse domains**: Covers multiple agent task domains and applications
- **LLM-as-Judge evaluation**: Uses language models for semantic assessment of answers
- **Large scale**: Extensive collection of agent trajectories with multi-turn interactions

## Dataset

- Located in: `test/open_end_qa_set.jsonl`
- **Evaluation method**: LLM-as-Judge accuracy assessment
- Questions require comprehensive understanding and recall of information from agent trajectories

## Data Format

Each data sample in the dataset follows this structure:

```json
{
  "episode_id": "string, the UUID of the data sample",
  "task": "string, task description",
  "domain": "string, agent task domain",
  "task_type": "string, task classification from AgentBench source",
  "source": "string, data source",
  "success": "boolean, whether the task was completed successfully",
  "num_turns": "integer, number of interaction turns",
  "total_tokens": "integer, total tokens used in the episode",
  "trajectory": [
    {
      "turn_idx": "integer, turn index",
      "action": "string, agent action",
      "observation": "string, environment observation"
    }
  ],
  "qa_pairs": [
    {
      "question": "string, memory-related question about the trajectory",
      "answer": "string, ground-truth answer",
      "question_uuid": "string, question unique id"
      "type": "string, question type, A: Recall; B: Causal Inference ; C: State Updating; D: State Abstraction"
    }
  ]
}
```

## Field Descriptions

### Top-level Fields

- **episode_id**: Unique identifier (UUID) for each data sample
- **task**: Natural language description of the agent's task
- **domain**: The application domain of the task
- **task_type**: Task category from the original AgentBench taxonomy
- **source**: Origin dataset or benchmark
- **success**: Boolean indicating whether the agent successfully completed the task
- **num_turns**: Total number of interaction turns in the trajectory
- **total_tokens**: Total token count consumed during task execution

### Trajectory Structure

- **trajectory**: Sequential list of agent-environment interactions
  - **turn_idx**: Index of the turn in the sequence
  - **action**: The action taken by the agent
  - **observation**: The observation returned by the environment

### QA Pairs Structure

- **qa_pairs**: List of questions testing memory retention about the trajectory
  - **question**: Memory-related question about the trajectory
  - **answer**: Ground-truth answer to the question
  - **question_uuid**: Globally unique identifier (UUID4) for the question
  - **type**: Question type category:
    - **A**: Recall - Direct information retrieval from trajectory
    - **B**: Causal Inference - Understanding cause-effect relationships
    - **C**: State Updating - Tracking state changes over time
    - **D**: State Abstraction - High-level understanding of agent state

## Usage

### Loading the Dataset

```python
import json

# Load Open-Ended QA dataset
openqa_data = []
with open('test/open_end_qa_set.jsonl', 'r') as f:
    for line in f:
        openqa_data.append(json.loads(line))
```

### Evaluation Methodology

Evaluate model responses using an LLM-as-Judge approach:

1. **Input**: Ground-truth answer and model-generated answer
2. **Criteria**: Semantic similarity and factual accuracy
3. **Assessment**: Consider both content accuracy and completeness
4. **Output**: Accuracy score indicating whether the model answer is correct

For detailed evaluation scripts and benchmarks, refer to the [main GitHub repository](https://github.com/AMA-Bench/AMA-Hub).

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{zhao2026amabenchevaluatinglonghorizonmemory,
      title={AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications},
      author={Yujie Zhao and Boqin Yuan and Junbo Huang and Haocheng Yuan and Zhongming Yu and Haozhou Xu and Lanxiang Hu and Abhilash Shankarampeta and Zimeng Huang and Wentao Ni and Yuandong Tian and Jishen Zhao},
      year={2026},
      eprint={2602.22769},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.22769},
}
```

## License

This dataset is released under the MIT License.