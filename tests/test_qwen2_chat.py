"""交互式命令行对话的历史管理测试。"""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from qwen2 import chat_loop


class TestQwen2Chat(unittest.TestCase):
    def test_history_is_kept_and_can_be_cleared(self):
        seen = []

        def reply(_model, _tokenizer, messages):
            seen.append([message.copy() for message in messages])
            return f"答复{len(seen)}", 128

        with patch("builtins.input", side_effect=["你好", "继续", "/clear", "新问题", "/exit"]):
            with patch("qwen2.generate_reply", side_effect=reply):
                with redirect_stdout(io.StringIO()):
                    chat_loop(None, None)

        self.assertEqual([message["role"] for message in seen[0]], ["system", "user"])
        self.assertEqual([message["role"] for message in seen[1]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(seen[1][2]["content"], "答复1")
        self.assertEqual([message["role"] for message in seen[2]], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
