import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st

from ancsim.signal.filterclasses import FilterSum_IntBuffer
import ancsim.signal.freqdomainfiltering as fdf




@given(st.integers(min_value=1, max_value=128), 
        st.integers(min_value=1, max_value=10), 
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10))
def test_freq_time_domain_convolution_is_equal(irLen, numIn, numOut, numBlocks):
    signal = np.random.standard_normal((numIn, numBlocks*irLen))
    ir = np.random.standard_normal((numIn, numOut, irLen))

    tdFilt = FilterSum_IntBuffer(ir)
    fdFilt = np.fft.fft(np.concatenate((ir, np.zeros_like(ir)), axis=-1), axis=-1).T
    initSig = np.random.standard_normal((numIn, irLen))
    tdFilt.process(initSig)
    fdInput = np.concatenate((initSig, signal), axis=-1)

    tdOut = np.zeros((numOut, irLen*numBlocks))
    fdOut = np.zeros((numOut, irLen*numBlocks))
    for i in range(numBlocks):
        tdOut[:,i*irLen:(i+1)*irLen] = tdFilt.process(signal[:,i*irLen:(i+1)*irLen])
        fdOut[:,i*irLen:(i+1)*irLen] = fdf.convolveSum(fdFilt, fdInput[:,i*irLen:(i+2)*irLen])

    assert(np.allclose(fdOut, tdOut))
    