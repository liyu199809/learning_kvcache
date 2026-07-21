import json
import re

with open("traj_data/refine_debug.jsonl") as f:
    data = json.loads(f.readline())

print("=" * 60)
print("【CASE 概览】")
print("=" * 60)
print(f"scenario: {data['scenario']}")
print(f"task_idx: {data['task_idx']}")
print(f"task: {data['task'][:100]}...")
print(f"max_rounds: {data['max_rounds']}")
print(f"actual_rounds: {data['rounds']}")
print(f"success: {data['success']}")
print(f"success_at_round: {data['success_at_round']}")
print(f"elapsed_sec: {data['elapsed_sec']}")
print()

rd = data["rounds_detail"][0]
print("=" * 60)
print("【Round 1 详情】")
print("=" * 60)
print(f"message_range: {rd['message_range']}")
print(f"student_steps: {rd['student_steps']}")
print(f"student_made_tool_call: {rd['student_made_tool_call']}")
print(f"student_error: {rd['student_error']}")
print(f"verify_reward: {rd['verify_reward']}")
print(f"verify_reward_type: {rd['verify_reward_type']}")
print(f"teacher_advice: {rd['teacher_advice']}")
print(f"teacher_rounds_used: {rd['teacher_rounds_used']}")
print(f"teacher_tool_calls_made: {rd['teacher_tool_calls_made']}")
print(f"teacher_replay_ok: {rd['teacher_replay_ok']}")
print(f"teacher_error: {rd['teacher_error']}")
print()

print(f"student_new_tool_calls ({len(rd['student_new_tool_calls'])}):")
for i, tc in enumerate(rd["student_new_tool_calls"]):
    print(f"  [{i}] tool={tc.get('tool_name','?')}, args={str(tc.get('arguments','?'))[:80]}")
print()

print("=" * 60)
print("【学生对话交互流】")
print("=" * 60)
conv = data["student_conversation"]
turn_num = 0
for i, msg in enumerate(conv):
    role = msg["role"]
    content = msg.get("content", "") or ""

    if role == "system":
        print(f"[msg {i}] SYSTEM  ({len(content)} chars)")
        print()
    elif role == "user":
        if "Available tools:" in content:
            print(f"[msg {i}] USER - tools list ({len(content)} chars)")
        elif "Tool response:" in content:
            is_error = "Error:" in content
            err_short = ""
            if is_error:
                m = re.search(r'Error:\s*(.+?)(?:\n|$)', content)
                if m:
                    err_short = m.group(1)[:100]
            print(f"[msg {i}] USER - tool_response, error={is_error} {err_short}")
        else:
            print(f"[msg {i}] USER - task/other ({len(content)} chars): {content[:100]}...")
        print()
    elif role == "assistant":
        turn_num += 1
        call_tags = content.count("<tool_call>")
        tool_names = re.findall(r'"tool_name":\s*"([^"]+)"', content)
        has_think = "<think>" in content or content.strip().startswith("</think>")

        print(f"[msg {i}] ASSISTANT Turn {turn_num}")
        print(f"  content_len: {len(content)}")
        print(f"  <tool_call> count: {call_tags}")
        print(f"  tool_names found: {tool_names if tool_names else 'none'}")
        print(f"  has_think_tags: {has_think}")
        # 展示开头和结尾
        lines = content.strip().split("\n")
        print(f"  first_line: {lines[0][:120]}")
        if len(lines) > 1:
            print(f"  last_line:  {lines[-1][:120]}")
        print()

print("=" * 60)
print("【关键观察】")
print("=" * 60)
print(f"1. 学生在第 1 轮就成功了 (verify_reward=1.0)，因此教师未介入")
print(f"2. 总共有 {turn_num} 个 assistant 消息，{rd['student_steps']} 个 step")
print(f"3. student_new_tool_calls 有 {len(rd['student_new_tool_calls'])} 个成功调用")
