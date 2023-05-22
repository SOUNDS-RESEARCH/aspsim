import pytest
import numpy as np
import aspsim.utilities as util
from hypothesis import given
import hypothesis.strategies as st


@given(
    st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
)
def test_calc_block_sizes_totalEqualToNumSamples(numSamples, block_size):
    startIdx = np.random.randint(0, block_size)
    sizes = util.calc_block_sizes(numSamples, startIdx, block_size)
    assert np.sum(sizes) == numSamples


@given(
    st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
)
def test_calc_block_sizes_maxValueEqualToBlockLength(numSamples, block_size):
    startIdx = np.random.randint(0, block_size)
    sizes = util.calc_block_sizes(numSamples, startIdx, block_size)
    assert np.max(sizes) <= block_size


@given(
    st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
)
def test_calc_block_sizes_noZeroValues(numSamples, block_size):
    startIdx = np.random.randint(0, block_size)
    sizes = util.calc_block_sizes(numSamples, startIdx, block_size)
    assert np.min(sizes) > 0


@given(
    st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
)
def test_calc_block_sizes_allMiddleValuesEqualToblock_size(numSamples, block_size):
    startIdx = np.random.randint(0, block_size)
    sizes = util.calc_block_sizes(numSamples, startIdx, block_size)
    print(sizes)
    if len(sizes) >= 3:
        assert np.allclose(sizes[1:-1], block_size)


@given(
    st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=100)
)
def test_calc_block_sizes_firstValueCorrect(numSamples, block_size):
    startIdx = np.random.randint(0, block_size)
    sizes = util.calc_block_sizes(numSamples, startIdx, block_size)
    assert sizes[0] == np.min((block_size - startIdx, numSamples))
