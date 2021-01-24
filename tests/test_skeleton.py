# -*- coding: utf-8 -*-

import pytest

from kilojoule.skeleton import fib

__author__ = "Jack Maddox"
__copyright__ = "Jack Maddox"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
