import itertools
import random
import unittest

from evalplus.sanitize import code_extract as reference_extract, sanitize as reference_sanitize
from sanitize_fast import code_extract, sanitize


class SanitizeTests(unittest.TestCase):
    def test_exhaustive_short_inputs_match_byte_for_byte(self):
        alphabet = ["", " ", "x = 1", "if True:", "    x = 2", "not python !", "# comment"]
        for count in range(1, 5):
            for lines in itertools.product(alphabet, repeat=count):
                text = "\n".join(lines)
                self.assertEqual(code_extract(text), reference_extract(text), repr(text))

    def test_random_inputs_match_byte_for_byte(self):
        rng = random.Random(42)
        alphabet = ["", "def f(x):", "    return x + 1", "print(f(1))", "```python", "```",
                    "# comment", "text:", "    pass", "x = (", "1", ")", "x = '''", "'''", "import math"]
        for _ in range(300):
            text = "\n".join(rng.choices(alphabet, k=rng.randint(1, 12)))
            self.assertEqual(code_extract(text), reference_extract(text), repr(text))
            self.assertEqual(sanitize(text, "f"), reference_sanitize(text, "f"), repr(text))

    def test_long_valid_code_and_open_fence(self):
        text = "def f(x):\n" + "    x += 1\n" * 2000 + "    return x"
        self.assertEqual(code_extract(text), text)
        self.assertEqual(code_extract("```python\n" + text), text)
        self.assertEqual(sanitize("```python\n" + text, "f"), text)

    def test_incomplete_and_ast_only_constructs(self):
        fixtures = [
            "x = 1\nfrom __future__ import annotations\ny = 2",
            "return 1\nx = 2", "break\nx = 2", "continue\nx = 2",
            "try:\n    x = 1\nexcept Exception:\n    pass",
            "x = (\n +\n 1\n)", "@foo\n@bar\ndef f():\n    return 1",
            "x = '''\ninvalid !\n'''", "if True:\n    x = '''\ninvalid !\n    '''",
            "x = \\\n    1", "The problem asks me to solve this.\nLet me think again.\nNo code here!",
        ]
        for text in fixtures:
            self.assertEqual(code_extract(text), reference_extract(text), repr(text))
            self.assertEqual(sanitize(text, "f"), reference_sanitize(text, "f"), repr(text))


if __name__ == "__main__":
    unittest.main()
