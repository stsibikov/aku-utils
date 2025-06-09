from typing import Sequence
import numpy as np


class BiasFunc:
    def __init__(self, point1, point2, a) -> None:
        '''
        Implementation of Barron 2020 bias function, may be slightly modified

        [Link to plot](https://www.desmos.com/calculator/cybgxv3yd0) of the function.

        Example
        ---
        ### Smoothing
        Suppose we want to create a flexible array smoothing function based on
        weighted moving average.

        We want to be able to input just one number, which represents how smooth do we
        want the array to be, and the bias function outputs the width of the window for
        the weighted moving average, and the weights for each of the elements in the window.

        We can do this using this class. We set the first point to (1, 1) and second to
        (6, 0) - the X of the second point corresponds to the maximum possible length
        of the window. We will be interpolating between this points with the bias function,
        and extracting 1st, 2nd, 3rd and so on weights by getting the value of the bias function
        at 1, 2, 3 and so on.

        The bias function will interpolate between (1, 1) and (6, 0), so this
        means that, at maximum, we will be using 5 elements in our window.

        Finally, we set alpha (`a`) to a number between 0 and 1 (both non-inclusive).
        Numbers closer to 0 means that the array will be smoother, numbers closer
        to 1 means the array will retain more of the original shape, and alpha
        equal to .5 means that the interpolation between (1, 1) and (6, 0) will be linear.

        This is implemented in WindowWeights (same module)
        '''
        self.point1 = point1
        self.point2 = point2
        self.a = a

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        forbidden = [0, 1]
        if value in forbidden:
            raise ValueError(f"Attribute `a` cannot be part of: {forbidden}")
        self._a = value

    def __call__(self, value: float | Sequence[float]) -> float | Sequence[float]:
        x0 = self.point1[0]
        y0 = self.point1[1]

        x1 = self.point2[0]
        y1 = self.point2[1]

        a = self.a

        # just see the link in the doc string to see the function
        res = ((y1 - y0) / (x1 - x0) * (value - x0)) / (
            (1 / a - 2) * (1 - (value - x0) / (x1 - x0)) + 1
        ) + y0
        return res


def get_window_weights(
    a: float = 0.5,
    n: int = 5,
    normalize: bool = True
) -> np.ndarray:
    '''
    Creates window weights by using a bias function

    Weights are decreasing and positive. Increasing `n`
    makes the weights smoother, ie the change between this and the
    next weight is smaller.

    Decreasing `a` will produce weights that, when applied to an array for
    a weighted moving average, will produce a smoother array.
    '''
    bias_func = BiasFunc(point1=(0, 1), point2=(n, 0), a=a)
    w : np.ndarray = bias_func.__call__(np.arange(n))  # type: ignore
    w = w[w > 0]

    if normalize:
        w = w / w.sum()

    return w
