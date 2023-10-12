from . import Method, Method2, Method3
from . import losses

method_paths = {
    "Method": Method,
    "Method2": Method2,
    "Method3": Method3
}

__all__ = ['method_paths']