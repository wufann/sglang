import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=900, suite="stage-c-test-large-8-gpu-amd")

KIMI_K2_MODEL_PATH = "moonshotai/Kimi-K2-Instruct"

GSM8K_ACCURACY_THRESHOLD = 0.85


class TestKimiK2MLACPMoriAMD(CustomTestCase):
    """tp=8, attn-cp=8 — MLA prefill CP + mori A2A on AMD."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K2_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-prefill-context-parallel",
            "--moe-a2a-backend",
            "mori",
            "--attention-backend",
            "fa3",
            "--mem-frac",
            "0.7",
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_DISPATCH_DTYPE"] = "bf16"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "128"
        env["MORI_SHMEM_MODE"] = "ISOLATION"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=32,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (kimi-k2-mla-cp-mori-amd)\n"
                f'{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
