"""交互式命令行对话的历史管理测试。"""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from qwen2 import assistant_end_tokens, chat_loop, new_user_prompt_tokens


class TestQwen2Chat(unittest.TestCase):
    def test_incremental_user_tokens_match_full_chat_template(self):
        from transformers import AutoTokenizer
        from qwen2 import MODEL_ID

        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        except OSError:
            self.skipTest("本地没有 Qwen2 tokenizer")
        first = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "17加25等于多少？只回答数字。"},
        ]
        next_user = {"role": "user", "content": "请再说一遍刚才的答案。只回答数字。"}
        incremental = (
            tokenizer.apply_chat_template(first, tokenize=True, add_generation_prompt=True)
            + tokenizer.encode("42", add_special_tokens=False)
            + assistant_end_tokens(tokenizer)
            + new_user_prompt_tokens(tokenizer, first[0], next_user)
        )
        full = tokenizer.apply_chat_template(
            [*first, {"role": "assistant", "content": "42"}, next_user],
            tokenize=True, add_generation_prompt=True)
        self.assertEqual(incremental, full)

    def test_history_is_kept_and_can_be_cleared(self):
        seen = []

        sessions = []

        def reply(_model, _tokenizer, messages, session, *, max_new_tokens):
            self.assertEqual(max_new_tokens, 12)
            seen.append([message.copy() for message in messages])
            sessions.append(session)
            return f"答复{len(seen)}", 128

        with patch("builtins.input", side_effect=["你好", "继续", "/clear", "新问题", "/exit"]):
            with patch("qwen2.generate_reply", side_effect=reply):
                with redirect_stdout(io.StringIO()):
                    chat_loop(None, None, max_new_tokens=12)

        self.assertEqual([message["role"] for message in seen[0]], ["system", "user"])
        self.assertEqual([message["role"] for message in seen[1]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(seen[1][2]["content"], "答复1")
        self.assertEqual([message["role"] for message in seen[2]], ["system", "user"])
        self.assertIs(sessions[0], sessions[1])
        self.assertIsNot(sessions[1], sessions[2])


if __name__ == "__main__":
    unittest.main()
