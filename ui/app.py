"""Streamlit front end: spec in, base vs fine-tuned Verilog out, with a Verilator compile check.

    uv run --extra ui --extra mlx streamlit run ui/app.py      # Apple Silicon
    uv run --extra ui --extra vllm streamlit run ui/app.py     # CUDA (Linux / WSL2)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.prompts import build_messages  # noqa: E402
from inference import BACKENDS, detect_backend, list_models, load_backend  # noqa: E402
from ui.common import compile_check  # noqa: E402

EXAMPLE_SPEC = (
    "A 4-bit synchronous up counter. On each rising clock edge, if reset is high the "
    "count goes to 0; otherwise, if enable is high, the count increments and wraps at 15."
)

st.set_page_config(page_title="Verilog Coder", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_backend(model: str, backend: str, quantized: bool):
    return load_backend(model, backend, quantized)


def render(column, model: str, backend: str, quantized: bool, messages, max_tokens: int):
    with column:
        st.subheader(model)
        try:
            engine = get_backend(model, backend, quantized)
        except Exception as exc:  # missing local build, server not running, ...
            st.error(f"Could not load {model} on {backend}: {exc}")
            return
        placeholder, text = st.empty(), ""
        for chunk in engine.stream(messages, max_tokens):
            text += chunk
            placeholder.markdown(f"```verilog\n{text}\n```")
        from inference.base import Generation

        code = Generation(text, engine.name, model).code
        placeholder.code(code, language="verilog")

        result = compile_check(code)
        if result is None:
            st.caption("Compile check skipped: install Verilator to enable it.")
        elif result.ok:
            st.success("Compiles cleanly with Verilator (lint)")
        else:
            st.error("Verilator lint failed")
            with st.expander("Verilator output"):
                st.code(result.log.strip()[:4000])
        st.download_button("Download .v", code + "\n", file_name=f"{model}.v", key=f"dl-{model}")


models = list(list_models())
with st.sidebar:
    st.header("Settings")
    detected = detect_backend()
    backend = st.selectbox("Backend", BACKENDS, index=BACKENDS.index(detected),
                           help=f"Auto-detected: {detected}")
    quantized = st.toggle("4-bit build", value=backend == "mlx",
                          help="MLX 4-bit on Apple Silicon, AWQ on vLLM")
    tuned = st.selectbox("Fine-tuned model", [m for m in models if m.startswith("ft-")] or models,
                         index=max(0, len([m for m in models if m.startswith("ft-")]) - 1))
    compare = st.toggle("Compare with base model", value=True)
    base = st.selectbox("Base model", [m for m in models if m.startswith("base-")] or models,
                        index=2, disabled=not compare)
    max_tokens = st.slider("Max new tokens", 256, 4096, 2048, step=256)

st.title("Verilog / SystemVerilog generator")
st.caption("Fine-tuned on OpenRTLSet. Describe a module; optionally give its interface.")

spec = st.text_area("Specification", EXAMPLE_SPEC, height=140)
interface = st.text_area("Module interface (optional)", placeholder="module counter(input clk, ...);", height=90)

if st.button("Generate", type="primary", disabled=not spec.strip()):
    messages = build_messages(spec, interface.strip() or None)
    columns = st.columns(2) if compare else [st.container()]
    render(columns[0], tuned, backend, quantized, messages, max_tokens)
    if compare:
        render(columns[1], base, backend, quantized, messages, max_tokens)
