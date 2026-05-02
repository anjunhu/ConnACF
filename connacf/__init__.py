# Package marker for connacf.
# When imported as a package (e.g. 'from connacf.macf...' from the parent dir),
# we guard the legacy bare imports that only work when CWD is connacf/.
try:
    from .agentverse import *
    from .dataset import *
    from .model import *
    from .props import *
    from dataset import *
    from trainer import *
except ImportError:
    pass
