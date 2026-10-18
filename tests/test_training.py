import pytest

from training.config import estimate_cost, list_runs, load_run


def test_grid_runs_resolve():
    for name in list_runs():
        run = load_run(name)
        assert run["learning_rate"] > 0
        if run["method"] != "full":
            assert run["lora_alpha"] == 2 * run["lora_r"]


def test_qlora_mirrors_lora():
    for i in (1, 2, 3):
        lora, qlora = load_run(f"L{i}"), load_run(f"Q{i}")
        keys = ("lora_r", "lora_alpha", "target_modules", "learning_rate")
        assert all(lora[k] == qlora[k] for k in keys)


def test_unknown_run():
    with pytest.raises(KeyError):
        load_run("nope")


def test_cost():
    assert estimate_cost("H100", 2, 1800) == 3.95


def test_prompt_tokens_masked():
    transformers = pytest.importorskip("transformers")
    from training.dataset import IGNORE_INDEX, tokenize_example

    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    row = {"description": "Invert x.", "ioheader": "module a(input x, output y);", "code": "module a; endmodule"}
    ex = tokenize_example(row, tok, 2048)
    supervised = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE_INDEX]
    text = tok.decode(supervised)
    assert "endmodule" in text and "Invert x." not in text
    assert text.rstrip().endswith("<|im_end|>")


def test_prose_interface():
    from training.dataset import prose_for, use_prose

    row = {"id": "1", "module": "inv", "ioheader": "module inv(input [3:0] a, output [3:0] y);"}
    assert prose_for(row) == "Module name: `inv`. Ports:\n- input `a` (4 bits)\n- output `y` (4 bits)"
    assert prose_for(row | {"ioheader": "module inv #(parameter W=4)(input [W-1:0] a, output y);"}) is None
    assert prose_for(row | {"ioheader": "module inv(input signed [3:0] a, output y);"}) is None
    flips = [use_prose(str(i), 0.3, 42) for i in range(2000)]
    assert flips == [use_prose(str(i), 0.3, 42) for i in range(2000)]  # deterministic
    assert 0.25 < sum(flips) / len(flips) < 0.35
    assert not any(use_prose(str(i), 0.0, 42) for i in range(100))
