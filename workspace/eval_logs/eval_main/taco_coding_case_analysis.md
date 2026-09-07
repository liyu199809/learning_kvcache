# TACO step67 代码评测逐题 case 分析

分析时间：2026-09-05（Asia/Shanghai）

对比模型：原生 `Qwen3.5-4B` 与 `TACO step67`。下面的结果均来自已保存的官方 scorer 逐题结果，不是人工猜测。

原始结果分别位于 `workspace/eval_logs/eval_main/qwen3.5-4b-base` 和 `workspace/eval_logs/eval_main/full-final-taco-step67`。

## 一、能做对，但没有在 16K token 内提交 final

### Case 1：LCB `abc341_a` — Print 341（easy）

任务：输入 `N`，输出 `10` 重复 `N` 次后再加一个 `1`。例如 `N=4` 输出 `101010101`。

- 原生模型：通过，核心实现为 `result = "10" * n + "1"`。
- TACO：`empty_final_answer`，`finish_reason=length`，最终代码长度为 0。

这道题不需要复杂算法，因此失败更符合“停止/提交行为”问题，而不是代码能力不足。

### Case 2：HumanEval/77 — `iscube`

任务：判断整数是否为某个整数的立方，包括负数和 0。

- 原生模型：base/plus 测试全部通过。
- TACO：`empty_final_answer`，`finish_reason=length`，base/plus 均失败。

### Case 3：MBPP/268 — `find_star_num`

任务：返回第 `n` 个星形数，核心公式只是：

```python
return 6 * n * (n - 1) + 1
```

- 原生模型：base/plus 全部通过。
- TACO：`empty_final_answer`，`finish_reason=length`，最终代码长度为 0。

这是另一个“题目极简单但仍未交代码”的例子。

## 二、提交了代码，但存在真实逻辑 bug

### Case 4：LCB `abc314_a` — 3.14（easy）

任务：输出圆周率到小数点后 `N` 位，不能删除末尾 0。

原生模型正确保留了 `3.`：

```python
result = pi_str[:n + 2]
```

TACO 却只截取小数数字：

```python
result = pi_str[2 : 2 + n]
```

因此 `N=2` 时它输出 `14`，而不是 `3.14`。这是明确的题意/输出格式理解错误。

### Case 5：HumanEval/37 — `sort_even`

任务：只对偶数下标上的值排序，奇数下标保持不变。

TACO 在交错构造结果后又执行：

```python
if len(l) % 2 == 1:
    result.append(l[-1])
```

但奇数长度列表的最后一个元素本来就在偶数下标，已经被循环加入。例如 `[1, 2, 3]` 会变成 `[1, 2, 3, 3]`。原生模型通过，TACO 的 base/plus 均失败。

### Case 6：MBPP/71 — `comb_sort`

TACO 的循环条件是：

```python
while gap > 0 or swapped:
    gap = int(gap / 1.3)
    if gap < 1:
        gap = 1
```

`gap` 被永久限制为至少 1，所以 `gap > 0` 永远为真，排序完成后仍无法退出循环。原生模型使用正确的：

```python
while gap > 1 or swapped:
```

### Case 7：LCB `abc351_f` — Double Sum（hard）

任务：计算所有 `i < j` 的 `max(A[j] - A[i], 0)`，需要坐标压缩和 Fenwick Tree。

TACO 第一遍从左到右统计后，直接复用了已填充的 `bit`进行反向统计，没有清零：

```python
# 缺少 bit = [0] * (m + 1)
right_count = 0
for x in reversed(a):
    ...
```

原生模型在反向遍历前正确重置了 BIT。TACO 会把左向统计残留与右向统计混在一起，导致错误结果。

## 三、反例：TACO 也确实救回了一些原生模型的空答案

### Case 8：LCB `2819` — remove trailing zeros（easy）

原生模型因长度上限没有提交 final；TACO 提交了并通过：

```python
class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        return num.rstrip('0')
```

### Case 9：HumanEval/147 — `get_max_triples`

原生模型是 `empty_final_answer`，TACO 成功利用模 3 分类和组合数公式，base/plus 全部通过。

### Case 10：MBPP/798 — array sum

原生模型是 `empty_final_answer`，TACO 直接提交并通过：

```python
def _sum(arr):
    return sum(arr)
```

## 总结

这些 case 同时说明了两件事：

1. TACO 并非完全没有学到代码能力，它会救回一部分原生模型的空答案。
2. 但净效果为负：LCB v5 上原生正确、TACO 空 final 有 93 题，而反向救回只有 57 题；另外还有 7 题从原生正确变成 TACO 的非空错误。

因此优先级应该是先修复“在 token budget 内结束 reasoning 并提交 final”，然后再处理少量真实算法/实现错误。
