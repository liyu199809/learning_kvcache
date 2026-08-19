import torch

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from prefix_tuning.virtual_prefix.modeling_qwen3_5_delta_virtual_prefix import (
    Qwen3_5DeltaVirtualPrefixGatedDeltaNet,
)


def main() -> None:
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        dtype=torch.bfloat16,
    )
    config.delta_prefix_num_virtual_tokens = 64
    module = Qwen3_5DeltaVirtualPrefixGatedDeltaNet(config, 0).cuda().bfloat16()
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(name == "prefix_tokens")
    with torch.no_grad():
        module.prefix_tokens.normal_(0.0, 0.02)

    lengths = [71, 39, 83]
    values = [
        torch.randn(1, length, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        for length in lengths
    ]
    weights = [
        torch.randn(1, length, config.hidden_size, device="cuda", dtype=torch.float32)
        for length in lengths
    ]

    reference_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    reference_outputs = [module(value, attention_mask=None) for value in reference_inputs]
    sum(
        (output.float() * weight).sum()
        for output, weight in zip(reference_outputs, weights, strict=True)
    ).backward()
    reference_prefix_grad = module.prefix_tokens.grad.detach().float().clone()
    reference_input_grad = torch.cat(
        [value.grad.detach().float() for value in reference_inputs], dim=1
    )
    module.zero_grad(set_to_none=True)

    packed_input = torch.cat(values, dim=1).detach().clone().requires_grad_(True)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    packed_output = module(
        packed_input,
        attention_mask=None,
        cu_seq_lens_q=cu_seqlens,
    )
    (packed_output.float() * torch.cat(weights, dim=1)).sum().backward()
    packed_prefix_grad = module.prefix_tokens.grad.detach().float()

    reference_output = torch.cat(reference_outputs, dim=1)
    output_error = (reference_output.float() - packed_output.float()).abs().detach()
    prefix_grad_error = (reference_prefix_grad - packed_prefix_grad).abs().detach()
    input_grad_error = (reference_input_grad - packed_input.grad.float()).abs().detach()
    metrics = {
        "output_max_abs": float(output_error.max()),
        "output_mean_abs": float(output_error.mean()),
        "prefix_grad_max_abs": float(prefix_grad_error.max()),
        "prefix_grad_mean_abs": float(prefix_grad_error.mean()),
        "input_grad_max_abs": float(input_grad_error.max()),
        "prefix_grad_finite": bool(torch.isfinite(packed_prefix_grad).all()),
        "prefix_grad_nonzero": bool(torch.count_nonzero(packed_prefix_grad)),
    }
    print(metrics)


if __name__ == "__main__":
    main()
