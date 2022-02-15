import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import ancsim.signal.filterclasses as fc
import ancsim.presets as preset
import pytest

import matplotlib.pyplot as plt
import time


# @pytest.fixture
# def setupconstants():
#     pos = preset.getPositionsCylinder3d()
#     sr = int(1000 + np.random.rand() * 8000)
#     noiseFreq = int(100 + np.random.rand() * 800)
#     return pos, sr, noiseFreq


def test_hardcoded_filtersum():
    ir = np.vstack((np.sin(np.arange(5)), np.cos(np.arange(5))))
    filt1 = fc.FilterSum(ir[:, None, :])

    inSig = np.array([[10, 9, 8, 7, 6, 5], [4, 5, 4, 5, 4, 5]])
    out = filt1.process(inSig)

    hardcodedOut = [
        [4.0, 15.57591907, 21.70313731, 17.44714985, 4.33911864, 3.58393245]
    ]
    assert np.allclose(out, hardcodedOut)


def test_impulseir_filtersum():
    filt1 = fc.FilterSum(np.ones((1, 1, 1)))

    ir2 = np.zeros((1, 1, 10))
    ir2[0, 0, 0] = 1
    filt2 = fc.FilterSum(ir2)

    inSig = np.random.rand(1, 16)

    out1 = filt1.process(inSig)
    out2 = filt2.process(inSig)

    assert np.allclose(out1, out2)


def test_impulseir_onedimfilter():
    filt1 = fc.Filter_IntBuffer(
        np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    )
    inSig = np.random.rand(1, 16)
    out1 = filt1.process(inSig)
    assert np.allclose(out1, inSig)


def test_hardcoded_onedimfilter():
    filt1 = fc.Filter_IntBuffer(np.sin(np.arange(5)))
    inSig = np.array([[10, 9, 8, 7, 6, 5]])
    out1 = filt1.process(inSig)
    print(out1)
    realOut = np.array(
        [[0, 8.41470985, 16.66621313, 16.3266448, 6.86673143, 5.7316455]]
    )
    # assert 0
    assert np.allclose(out1, realOut)


@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=10),
)
def test_filtersum_dynamic_static_equal_results(irLen, numIn, numOut, numBlocks):
    rng = np.random.default_rng()
    ir = rng.normal(0, 1, size=(numIn,numOut,irLen))

    filt_dyn = fc.FilterSumDynamic(ir)
    filt = fc.FilterSum(ir)

    out = np.zeros((numOut, numBlocks * irLen))
    out_dyn = np.zeros((numOut, numBlocks * irLen))

    sig = rng.normal(0, 1, (numIn, numBlocks * irLen))
    for i in range(numBlocks):
        out[:, i * irLen : (i + 1) * irLen] = filt.process(
            sig[:, i * irLen : (i + 1) * irLen]
        )
        out_dyn[:, i * irLen : (i + 1) * irLen] = filt_dyn.process(
            sig[:, i * irLen : (i + 1) * irLen]
        )
    #print(np.mean(np.abs(tdOut - fdOut)**2))
    assert np.allclose(out, out_dyn)








@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=10),
)
def test_freq_time_filter_sum_equal_results(irLen, numIn, numOut, numBlocks):
    ir = np.random.standard_normal((numIn, numOut, irLen))
    # ir = np.zeros((numIn, numOut, irLen))
    # ir[:,:,0] = 1
    tdFilt = fc.FilterSum(ir=ir)
    fdFilt = fc.FilterSum_Freqdomain(ir=ir)

    tdOut = np.zeros((numOut, numBlocks * irLen))
    fdOut = np.zeros((numOut, numBlocks * irLen))

    signal = np.random.standard_normal((numIn, numBlocks * irLen))
    for i in range(numBlocks):
        fdOut[:, i * irLen : (i + 1) * irLen] = fdFilt.process(
            signal[:, i * irLen : (i + 1) * irLen]
        )
        tdOut[:, i * irLen : (i + 1) * irLen] = tdFilt.process(
            signal[:, i * irLen : (i + 1) * irLen]
        )
    assert np.allclose(tdOut, fdOut)

@hyp.settings(deadline=None)
@hyp.given(
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=3),
    st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
    st.integers(min_value=1, max_value=3),
)
def test_freq_time_md_filter_equal_results(irLen, dataDim, filtDim, numBlocks):
    ir = np.random.standard_normal((*filtDim, irLen))
    tdFilt = fc.FilterMD(dataDim, ir)
    fdFilt = fc.FilterMD_Freqdomain(dataDim, ir=ir)

    tdOut = np.zeros((*filtDim, dataDim, numBlocks * irLen))
    fdOut = np.zeros((*filtDim, dataDim, numBlocks * irLen))

    signal = np.random.standard_normal((dataDim, numBlocks * irLen))
    for i in range(numBlocks):
        fdOut[..., i * irLen : (i + 1) * irLen] = fdFilt.process(
            signal[..., i * irLen : (i + 1) * irLen]
        )
        tdOut[..., i * irLen : (i + 1) * irLen] = tdFilt.process(
            signal[..., i * irLen : (i + 1) * irLen]
        )
    assert np.allclose(tdOut, fdOut)


# def test_same_output():
#     numIn = 10
#     numOut = 10
#     irLen = 4096
#     sigLen = 4096
#     ir = np.random.standard_normal((numIn, numOut, irLen))

#     sig = np.random.standard_normal((numIn, sigLen))
#     newFilt = fcn.FilterSum(ir = ir)
#     oldFilt = FilterSum(ir = ir)

#     s = time.time()
#     newOut = newFilt.process(sig)
#     print("new algo: ", time.time()-s)
#     s = time.time()
#     oldOut = oldFilt.process(sig)
#     print("old algo: ", time.time()-s)
#     #assert np.allclose(newOut, oldOut)
#     assert False

# def test_incremental_filtering():
#     numIn = 5
#     numOut = 5
#     irLen = 1024
#     sigLen = 1024
#     ir = np.random.standard_normal((numIn, numOut, irLen))
#     sig = np.random.standard_normal((numIn, sigLen))

#     incrFilt = fcn.FilterSum(ir = ir)
#     fullFilt = fcn.FilterSum(ir = ir)

#     incrementalOut = np.zeros((numOut, sigLen))
#     fullOut = np.zeros((numOut, sigLen))

#     s = time.time()
#     for i in range(sigLen):
#         incrementalOut[:,i:i+1] = incrFilt.process(sig[:,i:i+1])
#     print("Incremental algo time: ", time.time()-s)

#     s = time.time()
#     fullOut[:,:] = fullFilt.process(sig)
#     print("Full algo time: ", time.time()-s)

#     assert np.allclose(fullOut, incrementalOut)
#     #assert False


# def test_filt_time():
#     numIn = 5
#     numOut = 5
#     irLen = 1024
#     sigLen = 1024
#     ir = np.random.standard_normal((numIn, numOut, irLen))
#     sig = np.random.standard_normal((numIn, sigLen))

#     newFilt = fcn.FilterSum(ir = ir)
#     oldFilt = FilterSum(ir = ir)

#     newOut = np.zeros((numOut, sigLen))
#     oldOut = np.zeros((numOut, sigLen))

#     s = time.time()
#     for i in range(sigLen):
#         oldOut[:,i:i+1] = oldFilt.process(sig[:,i:i+1])
#     print("old algo time: ", time.time()-s)

#     s = time.time()
#     for i in range(sigLen):
#         newOut[:,i:i+1] = newFilt.process(sig[:,i:i+1])
#     print("new algo time: ", time.time()-s)
#     #assert np.allclose(oldOut, newOut)
#     assert False
