from setuptools import setup


setup(
    name="qwen35-delta-virtual-prefix-vllm",
    version="0.2.0",
    description="vLLM registry plugin for Qwen3.5 per-layer Delta virtual tokens",
    packages=["prefix_tuning", "prefix_tuning.virtual_prefix"],
    package_dir={"prefix_tuning": "."},
    entry_points={
        "vllm.general_plugins": [
            "qwen35_delta_virtual_prefix=prefix_tuning.virtual_prefix.vllm_plugin:register",
        ]
    },
)
