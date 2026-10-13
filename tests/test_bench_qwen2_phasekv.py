"""真实缓存基准的题目与评分边界测试，不加载模型。"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.bench_qwen2_phasekv import (
    answer_hit,
    full_sequence_token_match,
    load_cases,
    repeated_fourgram_fraction,
)


class TestQwen2Benchmark(unittest.TestCase):
    def test_short_prefix_is_not_full_token_agreement(self):
        self.assertEqual(full_sequence_token_match([1, 2], [1, 2, 3, 4]), 0.5)

    def test_repetition_and_numeric_answer_boundary(self):
        self.assertGreater(repeated_fourgram_fraction([1, 2, 3, 4, 1, 2, 3, 4]), 0)
        self.assertFalse(answer_hit("编号是1420", ["42"]))
        self.assertTrue(answer_hit("答案是 42。", ["42"]))

    def test_jsonl_prompt_is_converted_to_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"id": "one", "prompt": "你好", "answers": ["您好"]}) + "\n",
                            encoding="utf-8")
            cases = load_cases(str(path))
        self.assertEqual(cases[0]["messages"][0]["role"], "system")
        self.assertEqual(cases[0]["messages"][1]["content"], "你好")


if __name__ == "__main__":
    unittest.main()
