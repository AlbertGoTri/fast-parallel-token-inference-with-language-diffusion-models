"""The Alpaca eval generator must emit assertions in the exact inline
`return judge(output, '<question>')` shape the pipeline renderer extracts and batches, with the
reference answer / instruction grounded into the capability questions. No dataset download here:
the generator's row->config logic is tested on fake rows."""

import importlib.util
import os
import re

import yaml

from LLaDA.nested_distillation_eval import create_promptfoo_config_for_round

_GEN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "LLaDA", "evaluation", "promptfoo", "generate_alpaca_eval.py")
_spec = importlib.util.spec_from_file_location("generate_alpaca_eval", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# The regex the renderer uses to pull the question out of each assertion (must stay in sync).
_EXTRACT = re.compile(r"return\s+judge\(output,\s*(?:\"|')(.+?)(?:\"|')\s*\)\s*$", re.M)

_ROWS = [
    {"instruction": "Write a haiku about \"autumn\".\nThree lines.", "input": "",
     "output": "Crimson leaves descend\nOnto the quiet stone path\nAutumn breathes farewell"},
    {"instruction": "Explain what a prime number is.", "input": "",
     "output": "A prime is a natural number > 1 divisible only by 1 and itself. " * 12},
    {"instruction": "Give a path like C:\\a\\b and quote 'it'.", "input": "",
     "output": "Here is 'quoted' text with a C:\\back\\slash path."},
]


def _wrapped_compiles(value):
    src = "def get_assert(output, context):\n" + re.sub(r"(?m)^", "  ", value)
    compile(src, "<assert>", "exec")


def test_five_assertions_each_extractable_and_compilable():
    tests = gen.build_tests(_ROWS, len(_ROWS))
    assert len(tests) == len(_ROWS)
    for t in tests:
        assert len(t["assert"]) == 5
        for a in t["assert"]:
            assert a["type"] == "python"
            m = _EXTRACT.search(a["value"])
            assert m, f"question not extractable:\n{a['value']}"
            _wrapped_compiles(a["value"])


def test_sanitize_removes_quotes_and_backslashes():
    dirty = "line one\nhas 'single' and \"double\" and C:\\back\\slash   spaces"
    clean = gen._sanitize(dirty, 500)
    assert "'" not in clean and '"' not in clean and "\\" not in clean
    assert "\n" not in clean
    assert "  " not in clean  # whitespace collapsed


def test_sanitize_truncates():
    assert gen._sanitize("x" * 1000, 100).endswith("...")
    assert len(gen._sanitize("x" * 1000, 100)) <= 104


def test_capability_questions_ground_reference_and_instruction():
    tests = gen.build_tests(_ROWS, 1)
    qs = [_EXTRACT.search(a["value"]).group(1) for a in tests[0]["assert"]]
    # Q4 grounds the reference answer, Q5 grounds the instruction.
    assert "Reference:" in qs[3] and "Crimson leaves descend" in qs[3]
    assert "Instruction:" in qs[4] and "haiku" in qs[4]


def test_round_trips_through_renderer(tmp_path):
    tests = gen.build_tests(_ROWS, len(_ROWS))
    config = {"prompts": ["{{prompt}}"],
              "providers": [{"id": "file://llada_api_provider.py", "label": "LLaDA Student"}],
              "tests": tests}
    tmpl = tmp_path / "tmpl.yaml"
    with open(tmpl, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True, width=1000000)
    out = tmp_path / "rendered.yaml"
    create_promptfoo_config_for_round(str(tmpl), str(out), "llada_api_provider.py",
                                      max_concurrency=8, judge_num_gpu=99, judge_num_predict=256)
    rendered = yaml.safe_load(open(out, encoding="utf-8"))
    assert len(rendered["tests"]) == len(_ROWS)
    for t in rendered["tests"]:
        for a in t["assert"]:
            _wrapped_compiles(a["value"])           # batched wrapper still valid python
            assert '"num_gpu": 99' in a["value"]    # judge GPU count injected
