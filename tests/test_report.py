from eval.report import mcnemar_exact, wilson


def test_mcnemar_and_wilson():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and round(hi - lo, 2) == 0.19
