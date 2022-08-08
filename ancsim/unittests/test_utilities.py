import pytest
import numpy as np
import ancsim.utilities as util
from hypothesis import given
import hypothesis.strategies as st


@given(
    st.integers(min_value=1, max_value=10000), st.integers(min_value=1, max_value=10000)
)
def test_calc_block_sizes_totalEqualToNumSamples(numSamples, blockSize):
    startIdx = np.random.randint(0, blockSize)
    sizes = util.calc_block_sizes(numSamples, startIdx, blockSize)
    assert np.sum(sizes) == numSamples


@given(
    st.integers(min_value=1, max_value=10000), st.integers(min_value=1, max_value=10000)
)
def test_calc_block_sizes_maxValueEqualToBlockLength(numSamples, blockSize):
    startIdx = np.random.randint(0, blockSize)
    sizes = util.calc_block_sizes(numSamples, startIdx, blockSize)
    assert np.max(sizes) <= blockSize


@given(
    st.integers(min_value=1, max_value=10000), st.integers(min_value=1, max_value=10000)
)
def test_calc_block_sizes_noZeroValues(numSamples, blockSize):
    startIdx = np.random.randint(0, blockSize)
    sizes = util.calc_block_sizes(numSamples, startIdx, blockSize)
    assert np.min(sizes) > 0


@given(
    st.integers(min_value=1, max_value=10000), st.integers(min_value=1, max_value=10000)
)
def test_calc_block_sizes_allMiddleValuesEqualToBlockSize(numSamples, blockSize):
    startIdx = np.random.randint(0, blockSize)
    sizes = util.calc_block_sizes(numSamples, startIdx, blockSize)
    print(sizes)
    if len(sizes) >= 3:
        assert np.allclose(sizes[1:-1], blockSize)


@given(
    st.integers(min_value=1, max_value=10000), st.integers(min_value=1, max_value=10000)
)
def test_calc_block_sizes_firstValueCorrect(numSamples, blockSize):
    startIdx = np.random.randint(0, blockSize)
    sizes = util.calc_block_sizes(numSamples, startIdx, blockSize)
    assert sizes[0] == np.min((blockSize - startIdx, numSamples))
