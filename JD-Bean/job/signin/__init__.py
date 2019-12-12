from .JCoin import Coin
from .JHome import Home
from .JOrder import Order
from .JBean import Bean

jobs_all = [Home, Bean, Coin, Order]

jobs_minus = [Coin, Order, Bean]

jobs = [item for item in jobs_all if item not in jobs_minus]

__all__ = ["jobs"]