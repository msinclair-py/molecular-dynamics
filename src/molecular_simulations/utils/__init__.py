from .amber_utils import assign_chainids

try:
    from .parsl_settings import (AuroraSettings,
                                 LocalSettings,
                                 WorkstationSettings,
                                 PolarisSettings)
except ImportError:
    pass
