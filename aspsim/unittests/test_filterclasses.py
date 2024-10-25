import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import aspsim.signal.filterclasses as fc


def test_hardcoded_filtersum():
    ir = np.vstack((np.sin(np.arange(5)), np.cos(np.arange(5))))
    filt1 = fc.create_filter(ir=ir[:, None, :])

    inSig = np.array([[10, 9, 8, 7, 6, 5], [4, 5, 4, 5, 4, 5]])
    out = filt1.process(inSig)

    hardcodedOut = [
        [4.0, 15.57591907, 21.70313731, 17.44714985, 4.33911864, 3.58393245]
    ]
    assert np.allclose(out, hardcodedOut)


@hyp.settings(deadline=None)
@hyp.given(
    ir_len = st.integers(min_value=1, max_value=8),
    num_samples = st.integers(min_value=1, max_value=32),
)
def test_filtersum_ending_zeros_does_not_affect_output(ir_len, num_samples):
    ir2 = np.zeros((1, 1, ir_len))
    ir2[0, 0, 0] = 1
    filt = fc.create_filter(ir2)

    in_sig = np.random.rand(1, num_samples)
    out = filt.process(in_sig)
    assert np.allclose(in_sig, out)