import shutil

import pytest

from eval.diffsim import diff_sim, rename_module

pytestmark = pytest.mark.skipif(shutil.which("verilator") is None, reason="needs verilator")

ADDER = "module top(input [7:0] a, input [7:0] b, output [8:0] s); assign s = a + b; endmodule"
ADDER_SV = """module top(input logic [7:0] a, input logic [7:0] b, output logic [8:0] s);
  always_comb s = {1'b0, a} + {1'b0, b};
endmodule"""
ADDER_BUG = "module top(input [7:0] a, input [7:0] b, output [8:0] s); assign s = a - b; endmodule"

COUNTER = """module top(input clk, input rst_n, input en, output reg [3:0] q);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) q <= 4'd0; else if (en) q <= q + 4'd1;
endmodule"""
COUNTER_SV = """module top(input logic clk, input logic rst_n, input logic en, output logic [3:0] q);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= '0;
    else q <= en ? q + 1'b1 : q;
  end
endmodule"""
COUNTER_BAD_RESET = COUNTER.replace("q <= 4'd0", "q <= 4'd5")


def test_equivalent_combinational():
    r = diff_sim(ADDER, ADDER_SV, "top")
    assert r.status == "pass", r.log
    assert r.samples == 500


def test_buggy_combinational():
    r = diff_sim(ADDER, ADDER_BUG, "top")
    assert r.status == "mismatch" and r.mismatches > 0 and "s:" in r.first_mismatch


def test_equivalent_sequential():
    r = diff_sim(COUNTER, COUNTER_SV, "top")
    assert r.status == "pass", r.log
    assert r.samples == 1000  # two samples per clock cycle


def test_wrong_reset_value():
    assert diff_sim(COUNTER, COUNTER_BAD_RESET, "top").status == "mismatch"


def test_compile_error():
    assert diff_sim(ADDER, "module top(input a; endmodule", "top").status == "compile_error"


def test_port_mismatch():
    narrow = ADDER.replace("output [8:0] s", "output [7:0] s").replace("a + b", "a + b")
    assert diff_sim(ADDER, narrow, "top").status == "port_mismatch"


def test_wrong_module_name():
    assert diff_sim(ADDER, ADDER.replace("module top", "module adder"), "top").status in (
        "compile_error", "port_mismatch")



def test_rename_only_touches_declaration():
    code = "module q(input a, output q_out); wire q; assign q = a; assign q_out = q; endmodule"
    renamed = rename_module(code, "q", "ref_q")
    assert renamed.startswith("module ref_q(") and "wire q;" in renamed
