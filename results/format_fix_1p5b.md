| Model | OpenRTLSet pass@1 | VerilogEval pass@1 | VerilogEval + interface | Compile (ORT / VE / VE+if) | vs base, ORT: +/- (p) | vs base, VE+if: +/- (p) | Train cost |
|---|---|---|---|---|---|---|---|
| base-Qwen2.5-Coder-1.5B-Instruct | 17.3 (68/394) | 10.3 (16/156) | 19.2 (30/156) | 48% / 51% / 60% | baseline | baseline | - |
| L3 | 39.9 (157/394) | 10.3 (16/156) | 26.9 (42/156) | 73% / 29% / 63% | +93/-4 (p=0.0) | +20/-8 (p=0.0357) | $1.03 |
| L3i | 40.6 (160/394) | 20.5 (32/156) | 25.0 (39/156) | 74% / 58% / 67% | +95/-3 (p=0.0) | +15/-6 (p=0.0784) | $1.03 |
