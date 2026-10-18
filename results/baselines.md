| Model | OpenRTLSet pass@1 | VerilogEval pass@1 | VerilogEval + interface | Compile (ORT / VE / VE+if) | vs base, ORT: +/- (p) | vs base, VE+if: +/- (p) | Train cost |
|---|---|---|---|---|---|---|---|
| base-Qwen2.5-Coder-1.5B-Instruct | 17.3 (68/394) | 10.3 (16/156) | 19.2 (30/156) | 48% / 51% / 60% | +3/-66 (p=0.0) | +1/-31 (p=0.0) | - |
| base-Qwen2.5-Coder-3B-Instruct | 25.6 (101/394) | 18.6 (29/156) | 23.1 (36/156) | 63% / 59% / 69% | +11/-41 (p=0.0) | +3/-27 (p=0.0) | - |
| base-Qwen2.5-Coder-7B-Instruct | 33.2 (131/394) | 31.4 (49/156) | 38.5 (60/156) | 82% / 74% / 85% | baseline | baseline | - |
| base-Qwen2.5-Coder-14B-Instruct | 37.6 (148/394) | 37.8 (59/156) | 44.2 (69/156) | 84% / 72% / 85% | +33/-16 (p=0.0213) | +14/-5 (p=0.0636) | - |
