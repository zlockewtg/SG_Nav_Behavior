# Shim: when the repo lives at SG-Nav/segment_anything/, running scripts from SG-Nav
# puts this directory on sys.path first; without this file Python treats it as an empty
# namespace package and shadows the real implementation in segment_anything/segment_anything/.
from .segment_anything import *  # noqa: F403
