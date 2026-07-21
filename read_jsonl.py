import json, copy
from transformers import AutoTokenizer

def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def normalize_tool_calls(messages):
    """把 assistant 消息里 tool_calls 的 arguments 从 JSON 字符串转成 dict，
    以兼容 Qwen3.5 chat template 的 `arguments|items` 要求。"""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)  # 兼容有无 function 包裹两种结构
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    fn["arguments"] = {}   # 解析失败兜底，避免模板再炸
    return msgs

data =  read_jsonl("traj_data/refine_debug.jsonl")

print(data[0])

tokenizer = AutoTokenizer.from_pretrained("/mnt/storage/disk1/verl_data/base_model/Qwen3.5-4B")
prompt = tokenizer.apply_chat_template(data[0]["student_conversation"], tokenize=False, add_generation_prompt=True)
