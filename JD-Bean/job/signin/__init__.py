from .JCoin import Coin
from .JHome import Home
from .JOrder import Order
from .JBean import Bean
from .JLike import Like

jobs_all = [Like, Home, Bean, Coin, Order]

# jobs_minus = [Coin, Order, Bean, Home]
jobs_minus = [Order,Like, Home,Coin]

jobs = [item for item in jobs_all if item not in jobs_minus]

__all__ = ["jobs"]