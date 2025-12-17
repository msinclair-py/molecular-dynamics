"""
Test configuration and fixtures.
"""
import os

# Disable numba JIT compilation to avoid path resolution issues during testing.
# This must be set before numba is imported.
os.environ['NUMBA_DISABLE_JIT'] = '1'
