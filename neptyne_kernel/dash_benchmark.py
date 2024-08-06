import time
from unittest import mock

from .dash import Dash
from .test_utils import a1


def benchmark_dash(dash):
    """
    Timings:
        before: 0.217
        after: 0.0955
    """
    t = time.time()
    for i in range(10):
        dash[a1("A1")] = [[ord(ch1) * 10 + i for ch1 in "ABCDEF"] for i in range(1000)]
        for row in dash[a1("A1:F1000")]:
            for j in range(len(row)):
                dash[a1("G1")] += row[j]

    print(time.time() - t)


if __name__ == "__main__":
    with mock.patch("neptyne_kernel.dash.get_ipython_mockable") as mock_get_ipython:
        benchmark_dash(Dash(silent=True))
