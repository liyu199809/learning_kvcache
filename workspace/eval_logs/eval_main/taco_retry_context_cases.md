# TACO “长推理失败 → reset/retry → 第二次提交”训练 case

分析时间：2026-09-05（Asia/Shanghai）

## 数据结构

TACO 训练 parquet 中有 851/1625 条是以下四轮上下文：

```text
system: 代码任务指令
user: 题目
assistant: 第一次失败/部分成功的长回答
user: Notice: Your first tried failed ... please try again.
```

其中 817 条是 `complete_after_refine`，34 条是 `improved_partial`。第一次 assistant 回答平均 41,696 字符，中位数 43,687，最长 117,166；851 条全部包含 thinking，只有 379 条包含 fenced Python 代码。

重要边界：这段失败回答是 **prompt context**，不直接计算 loss。训练时学生模型要在 retry 之后生成新的第二次回答；teacher 额外看到 `teacher_prompt` 中的参考解法/建议，并对学生的第二次回答做 on-policy distillation。

## Case 1：`taco_4690` — 最大数位和

题目只有 438 字符：求不大于 `N` 的正整数中，最大可能的十进制数位和。

### 第一次回答

- 43,612 字符。
- `source_score=0.0`。
- 整段都在 `<think>...</think>` 中，`</think>` 前仍停在未完成的论证，后面没有 final 代码。
- 中间曾经写出看似正确的代码，但又继续反复重证 `N=10`、`N=50` 等 case，最后在另一轮论证中截断。

随后 prompt 追加 reset/retry 消息，teacher 获得一份 1,369 字符的通过解法，数据构建记录为 `target_score=1.0`。

### 训练时的第二次回答

该题出现在 `rollout_trajs/.../59.jsonl`：

- 第二次回答仍长达 27,681 字符。
- 但这次提交了代码，103/103 测试全过，`score=1.0`。
- 回答中明确出现了 `reference solution` 话术，尽管学生的可见 input 中并没有参考答案。

这是最直接的“第一次长推理没交代码 → retry → 第二次长推理后正确提交” case。

## Case 2：`taco_2454` — 连续平方数列

题目只有 267 字符：实现 `squares(x, n)`，返回 `[x, x², x⁴, ...]`。

### 第一次回答

- 37,082 字符，其中约 36,228 字符是 thinking。
- 大量反复纠结参数顺序。
- 最后提交了 `generate_sequence(x, n)`，而测试需要的 entrypoint 是 `squares`，因此 `source_score=0.0`。
- retry 后 teacher 可见参考解法使用正确的 `def squares(x, n)`，数据记录 `target_score=1.0`。

### 训练时的第二次回答

该题出现在 step 15：

- 第二次回答缩短到 3,263 字符。
- 但它仍然提交 `generate_sequence`，6/6 测试全部 runtime error，`score=0.0`。

这表明数据行的设计意图是“第二次纠错”，但 on-policy 训练中学生当时生成的第二次回答并不保证真的纠正。

## Case 3：`taco_2574` — Ninja House

这是最极端的 case：

- 第一次 assistant 回答 117,166 字符，全是 thinking，末尾是大量反复的 `Wait`，没有 final 代码，`source_score=0.0`。
- reset/retry 后，step 33 的第二次回答仍长达 101,674 字符。
- 第二次回答仍在反复 `Wait`，最终是 `format_error`，0/101 测试通过。

这说明“超长失败上下文 → 再试一次”不仅存在，还可能让第二次回答继续复制同样的失控模式。

## Case 4：`taco_5715` — URL 域名解析

- 题目 309 字符。
- 第一次回答 40,355 字符，提交了代码，但无法处理不带 scheme 的 URL，通过 7/9，`source_score=0.7778`。
- retry 时 teacher 看到的 expert advice 明确指出：`urlparse` 会把无 scheme 的 hostname 放进 `path`，应回退到 `parsed.path` 或先补 `http://`。
- step 30 的第二次回答改成了错误正则，反而 0/9 通过。

这是“retry 后不仅未纠正，反而更差”的反例。

## 结论

“先长推理，失败后再纠错”不是抽象推测，而是 TACO 训练数据中 851/1625 条样本的显式对话结构。

但更准确的表述应该是：

> 数据使学生频繁在“已有一段超长失败回答 + reset/retry”的条件下学习第二次回答；第一次失败文本本身不计 loss，但训练分布与评测时的干净首轮 prompt 明显不同。
