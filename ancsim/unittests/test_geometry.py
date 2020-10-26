import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from ancsim.soundfield.geometryutilities import getCenterOfCircle
from ancsim.soundfield.generatepoints import equiangularCircle


# CURRENTLY NOT BEING TESTED;
# ADD TEST TO START OF FUNCTION NAME TO READD IT TO TEST LISTS
@given(
    st.integers(min_value=3, max_value=100),
    st.floats(min_value=0.1, max_value=100),
    st.floats(min_value=-1000, max_value=1000),
    st.floats(min_value=-1000, max_value=1000),
)
def center_single_circle(numPoints, rad, xCenter, yCenter):
    pos = equiangularCircle(numPoints, (rad, rad))
    # if numPoints == 3:
    #    print(pos)
    pos[:, 0] += xCenter
    pos[:, 1] += yCenter

    center, radius = getCenterOfCircle(pos)
    assert (
        np.abs(center[0, 0] - xCenter) < 1e-10
        and np.abs(center[0, 1] - yCenter) < 1e-10
    )
