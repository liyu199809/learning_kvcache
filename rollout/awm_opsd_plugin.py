"""AWM OPSD dataset plugin + verify reward for ms-swift.

Two pieces:

1. OPSD dataset preprocessor (`AWMOPSDBatchPreprocessor`): converts a
   resample JSONL (already joined with teacher advice via rollout.join_advice)
   into swift's training format.

   OPSD requires teacher and student to score the SAME token sequence, with
   the teacher only differing by extra privileged context. So:
     - `messages`:       the student's own conversation (NO advice) — the
                         sequence the student actually generated.
     - `teacher_prompt`: the SAME conversation with the advice prepended to
                         the system prompt — the teacher's privileged view of
                         the identical sequence.

   Tool-call results (role == "tool") are masked from loss automatically by
   swift's template (loss is computed on assistant turns only), so the model
   never learns to predict environment responses.

2. Verify reward (`AWMVerifyORM`): scores the student's final answer by the
   `verify_reward_type` that rollout.resample already computed on the live
   AWM server. GRPO consumes this pre-computed per-sample reward — no live
   env calls needed during training.

Before training, join advice into the resample JSONL:
    python -m rollout.join_advice \
        --resample-jsonl traj_data/resample_g16.jsonl \
        --refine-jsonl traj_data/refine_200.jsonl \
        --output-jsonl traj_data/resample_g16_with_advice.jsonl

Usage (GKD + OPSD, offline — recommended for the current 22-case set):
    swift rlhf --rlhf_type gkd \
        --external_plugins rollout/awm_opsd_plugin.py \
        --dataset traj_data/resample_g16_with_advice.jsonl ...

Usage (OPD-RL + OPSD, GRPO with verify reward):
    swift rlhf --rlhf_type grpo \
        --external_plugins rollout/awm_opsd_plugin.py \
        --reward_funcs awm_verify \
        --teacher_model qwen3.5-4b \
        --dataset traj_data/resample_g16_with_advice.jsonl ...
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.rewards import ORM, orms

# ---------------------------------------------------------------------------
# 1. OPSD dataset preprocessor
# ---------------------------------------------------------------------------
_ADVICE_SYSTEM_BLOCK = (
    "\n\n# Expert advice for this exact task\n"
    "A stronger model attempted this task before. Its advice (quoted below) "
    "points out the likely pitfall and the workflow that works.\n\n"
    "# Advice: {advice}"
)


class AWMOPSDBatchPreprocessor(RowPreprocessor):
    """Build swift rows from a resample JSONL record (advice already joined).

    Student view (`messages`): the resampled conversation verbatim — system +
    task + tool history + env-reset note + the model's own continuation. No
    privileged advice.

    Teacher view (`teacher_prompt`): the SAME conversation, with the advice
    appended to the system prompt. Teacher and student thus score the same
    token sequence; the teacher only sees extra privileged context.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conv = row.get("student_conversation")
        if not conv:
            return None
        advice = row.get("advice")
        if not advice:
            # Advice must be joined in beforehand (see module docstring).
            return None

        # Teacher conversation = student conversation with advice in system.
        teacher_conv = copy.deepcopy(conv)
        advice_block = _ADVICE_SYSTEM_BLOCK.format(advice=advice.strip())
        if teacher_conv and teacher_conv[0].get("role") == "system":
            teacher_conv[0]["content"] = (
                (teacher_conv[0].get("content") or "") + advice_block)
        else:
            teacher_conv.insert(0, {
                "role": "system",
                "content": advice_block.strip(),
            })

        return {
            "messages": conv,
            "teacher_prompt": teacher_conv,
            "scenario": row.get("scenario"),
            "task_idx": row.get("task_idx"),
            "group_idx": row.get("group_idx"),
            "verify_reward_type": row.get("verify_reward_type"),
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id="awm_opsd_resample",
        preprocess_func=AWMOPSDBatchPreprocessor(),
        tags=["awm", "opsd", "agent"],
    ))


# ---------------------------------------------------------------------------
# 2. AWM verify reward (ONLINE — from the student's own on-policy rollout)
# ---------------------------------------------------------------------------
class AWMVerifyORM(ORM):
    """Reward from the student's OWN on-policy rollout verify result.

    The AWM scheduler (awm_scheduler_plugin.AWMScheduler) calls the live AWM
    `verify` tool on the student's ACTUAL final answer at episode end and stores
    the result in ``rollout_infos['total_reward']`` (1.0 for "complete", else
    0.0). swift surfaces each sample's ``rollout_infos`` to reward functions as a
    batched column (GRPOSample.to_reward_row -> rows_to_batched), so we read the
    real per-sample reward here.

    Previously this read the static ``verify_reward_type`` dataset column. That
    column is built from successful teacher trajectories, so it is ALWAYS
    "complete" -> reward was constant 1.0 -> every GRPO group had zero variance
    -> advantages all 0 -> no learning signal. We now score the student's live
    rollout instead.
    """

    def __call__(self, completions, rollout_infos=None, **kwargs) -> List[float]:
        # swift calls reward_func(completions, **reward_kwargs); `completions`
        # MUST be the first positional param. `rollout_infos` arrives as a
        # batched column (one dict per completion) carrying the scheduler's live
        # verify result under 'total_reward'. Absent/missing (e.g. env init
        # failure, or a trajectory that never terminated normally) => 0.0, i.e.
        # "not complete".
        if not rollout_infos:
            return [0.0 for _ in completions]
        return [float((ri or {}).get("total_reward", 0.0)) for ri in rollout_infos]


orms["awm_verify"] = AWMVerifyORM


# ---------------------------------------------------------------------------
# 3. OPSD teacher-view loss-mask fix (swift bug workaround)
# ---------------------------------------------------------------------------
# swift's OPSD teacher view drops the on-policy response's `response_loss_mask`:
#   * student encode  (rlhf_trainers/utils.py:encode_sample) reads the mask via
#     `sample.response_loss_mask` directly.
#   * teacher encode  (rlhf_trainers/gkd_helpers.py:encode_teacher_view) reads it
#     via `teacher_row.get('response_loss_mask')`, but
#     `OnPolicySample.to_teacher_template_dict()` never puts it in that dict — so
#     the teacher always gets loss_mask=None.
#
# For Qwen3.5 (non_thinking_prefix='<think>\n\n</think>\n\n'), the on-policy
# response begins with that 4-token empty-think prefix, which the rollout marks
# loss=0. The student honors it; the teacher (mask=None) counts those 4 tokens
# as loss => teacher completion length = student + 4, tripping the OPSD invariant
#   "OPSD response length mismatch: student=131 teacher=135" (gkd_helpers.py:468).
#
# Fix: make `to_teacher_template_dict()` carry `response_loss_mask` so swift's own
# encode_teacher_view applies the SAME mask as the student. Kept as a plugin-side
# patch to leave the ms-swift kernel pristine (upstreamable one-liner).
def _install_opsd_loss_mask_fix() -> None:
    try:
        from swift.rl_core.data import OnPolicySample
    except Exception as e:  # noqa: BLE001
        print(f"[OPSD FIX] could not patch to_teacher_template_dict: {e!r}")
        return

    if getattr(OnPolicySample, "_awm_loss_mask_patched", False):
        return

    _orig_ttd = OnPolicySample.to_teacher_template_dict

    def _patched_ttd(self):
        d = _orig_ttd(self)
        # Carry the response loss mask into the teacher row so encode_teacher_view
        # masks the same tokens (e.g. the empty <think> prefix) the student masks.
        if self.response_token_ids and self.response_loss_mask:
            d["response_loss_mask"] = self.response_loss_mask
        return d

    OnPolicySample.to_teacher_template_dict = _patched_ttd
    OnPolicySample._awm_loss_mask_patched = True
    print("[OPSD FIX] patched to_teacher_template_dict to carry response_loss_mask")


_install_opsd_loss_mask_fix()


# ---------------------------------------------------------------------------
# 4. OPSD length-mismatch diagnostics (opt-in via OPSD_DEBUG=1)
# ---------------------------------------------------------------------------
# The OPSD invariant (swift rlhf_trainers/gkd_helpers.py:remap_teacher_logps_to_
# student_frame) requires the teacher view and the student view to have the SAME
# number of loss-bearing response tokens (labels != -100). If they differ, swift
# raises "OPSD response length mismatch at sample i: student=.. teacher=..".
#
# This wraps swift's `encode_teacher_view` so that, for every sample it encodes,
# it ALSO encodes the student view and — on a label-count mismatch — decodes and
# prints exactly which tokens diverge. Read-only w.r.t. the sample; it just
# returns the original teacher encoding. Enable with OPSD_DEBUG=1.
def _install_opsd_debug() -> None:
    import os
    if not os.environ.get("OPSD_DEBUG"):
        return
    try:
        import swift.rlhf_trainers.grpo_trainer as _gt
        from swift.rlhf_trainers.utils import encode_sample as _encode_sample
    except Exception as e:  # noqa: BLE001
        print(f"[OPSD DEBUG] could not hook encode_teacher_view: {e!r}")
        return

    _orig = _gt.encode_teacher_view

    def _as_list(x):
        try:
            return x.tolist()
        except AttributeError:
            return list(x)

    def _label_ids(enc):
        labels = _as_list(enc.get("labels") or [])
        ids = _as_list(enc.get("input_ids") or [])
        return [i for i, l in zip(ids, labels) if l != -100]

    def _debug_encode_teacher_view(sample, template):
        t_enc = _orig(sample, template)
        try:
            s_enc = _encode_sample(sample, template)
            t_lab, s_lab = _label_ids(t_enc), _label_ids(s_enc)
            if len(t_lab) != len(s_lab):
                tok = template.tokenizer
                print("=" * 70)
                print(f"[OPSD DEBUG] LABEL COUNT MISMATCH: "
                      f"student={len(s_lab)} teacher={len(t_lab)} "
                      f"(diff={len(t_lab) - len(s_lab)})")

                # Structural dump: roles + content type of BOTH message views.
                def _dump_msgs(tag, msgs):
                    print(f"--- {tag} messages ({len(msgs or [])} turns) ---")
                    for j, m in enumerate(msgs or []):
                        c = m.get("content")
                        if isinstance(c, dict):
                            ct = f"ids[{len(c.get('token_ids') or c.get('input_ids') or [])}]"
                        elif isinstance(c, list):
                            ct = f"ids[{len(c)}]"
                        else:
                            s = str(c)
                            ct = f"str<{s[:40]!r}...>" if len(s) > 40 else f"str<{s!r}>"
                        print(f"    [{j}] {m.get('role')}: {ct}")
                _dump_msgs("STUDENT", sample.messages)
                _dump_msgs("TEACHER", sample.teacher_messages)
                print(f"[OPSD DEBUG] response_token_ids type: "
                      f"{type(sample.response_token_ids).__name__} "
                      f"outer_len={len(sample.response_token_ids or [])} "
                      f"nested={isinstance((sample.response_token_ids or [None])[0], list)}")
                print(f"[OPSD DEBUG] chat_template_kwargs={sample.extra.get('chat_template_kwargs')}")

                print("--- student LOSS tokens (decoded) ---")
                print(repr(tok.decode(s_lab)))
                print("--- teacher LOSS tokens (decoded) ---")
                print(repr(tok.decode(t_lab)))
                n = min(len(s_lab), len(t_lab))
                div = next((k for k in range(n) if s_lab[k] != t_lab[k]), None)
                if div is not None:
                    print(f"[OPSD DEBUG] first loss-token divergence @ {div}: "
                          f"student={s_lab[div]}({tok.decode([s_lab[div]])!r}) "
                          f"teacher={t_lab[div]}({tok.decode([t_lab[div]])!r})")
                    print("  student ctx:",
                          repr(tok.decode(s_lab[max(0, div - 3):div + 5])))
                    print("  teacher ctx:",
                          repr(tok.decode(t_lab[max(0, div - 3):div + 5])))
                elif len(t_lab) > len(s_lab):
                    print("[OPSD DEBUG] teacher EXTRA leading loss tokens:",
                          repr(tok.decode(t_lab[:len(t_lab) - len(s_lab)])))
                    print("[OPSD DEBUG] teacher EXTRA trailing loss tokens:",
                          repr(tok.decode(t_lab[len(s_lab):])))
                else:
                    print("[OPSD DEBUG] student EXTRA trailing loss tokens:",
                          repr(tok.decode(s_lab[len(t_lab):])))
                print("=" * 70)
        except Exception as e:  # noqa: BLE001 — diagnostics must never break training
            import traceback
            print(f"[OPSD DEBUG] compare failed: {e!r}")
            traceback.print_exc()
        return t_enc

    _gt.encode_teacher_view = _debug_encode_teacher_view
    print("[OPSD DEBUG] patched encode_teacher_view for length-mismatch diagnostics")


_install_opsd_debug()
