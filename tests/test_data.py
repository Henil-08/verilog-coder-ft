from data.filters import categorize, dedup_key, is_standalone, module_name
from data.prompts import build_training_messages, extract_code
from data.split import allocate_test_quotas


def test_categorize():
    assert categorize("module a(input x, output y); assign y = x; endmodule") == "verilog"
    assert categorize("module a(input logic x); always_comb begin end endmodule") == "systemverilog"
    assert categorize("class t extends uvm_test; endclass") == "uvm"
    # keywords inside comments don't count
    assert categorize("// uses logic later\nmodule a; endmodule") == "verilog"


def test_dedup_ignores_comments_and_whitespace():
    a = "module a;\n  wire x; // note\nendmodule"
    b = "module a; wire x; /* other */ endmodule"
    assert dedup_key(a) == dedup_key(b)


def test_module_name_and_standalone():
    assert module_name("module top #(parameter W=8) (input clk);") == "top"
    assert is_standalone("module a; endmodule", has_children=False)
    assert not is_standalone("module a; endmodule\nmodule b; endmodule", has_children=False)
    assert not is_standalone("`include \"defs.vh\"\nmodule a; endmodule", has_children=False)
    assert not is_standalone("module a; endmodule", has_children=True)


def test_quotas_floor_small_categories():
    quotas = allocate_test_quotas({"verilog": 50000, "systemverilog": 4000, "uvm": 3}, 500, 50)
    assert quotas == {"systemverilog": 50, "uvm": 1, "verilog": 449}
    assert sum(quotas.values()) == 500


def test_prompt_roundtrip():
    row = {"description": "Invert x.", "ioheader": "module a(input x, output y);", "code": "module a; endmodule"}
    msgs = build_training_messages(row)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert extract_code(msgs[-1]["content"]) == row["code"]
    assert extract_code("no fence here") == "no fence here"
