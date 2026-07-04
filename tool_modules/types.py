# -*- coding: utf-8 -*-
"""通用类型定义模块。"""

from typing import List, Union

import numpy as np

ArrayLike = Union[np.ndarray, List[List[int]]]
