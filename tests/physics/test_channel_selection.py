"""
Return selected channels for each test case to make tests run consistently and without
user interaction required
"""
from numbers import Number
from typing import Callable
from typing import Collection
from typing import Iterable

import pytest
from xarray import DataArray

DataSelector = Callable[
    [DataArray, str, Collection[Number], Iterable[Number]], Iterable[Number]
]


@pytest.mark.skip("Not a test to run, used by physics tests")
def test_case_selector(
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number],
    unselected_channels: Iterable[Number] = None,
) -> Iterable[Number]:
    """
    Return channels from cache with no modification/input. To be used with
    custom cached channels for consistent selection
    """
    if unselected_channels is None:
        unselected_channels = []
    return list(unselected_channels)
