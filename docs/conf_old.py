# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'molecular-simulations'
copyright = '2025, Matt Sinclair'
author = 'Matt Sinclair'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown
# here.

# make sure sphinx always uses the current branch
from numbers import Real
import os
import sphinx_rtd_theme
import sys
import types

sys.path.insert(0, os.path.abspath('../src'))

# type shims for mocked imports
# --- start shim ---
omm = types.ModuleType('openmm')
unit = types.ModuleType('openmm.unit')
app = types.ModuleType('openmm.app')
app_internal = types.ModuleType('openmm.app.internal')

omm.__path__ = []
app.__path__ = []
app_internal.__path__ = []

class _Quantity:
    """Minimal Quantity stand-in that tolerates algebra at import time."""
    def __init__(self, mag=1): self.mag = mag
    # arithmetic with Real or other stand-ins -> always return a Quantity
    def _m(self, other): return other if isinstance(other, Real) else getattr(other, 'mag', 1)
    def __mul__(self, other):      return _Quantity(self.mag * self._m(other))
    def __rmul__(self, other):     return _Quantity(self._m(other) * self.mag)
    def __truediv__(self, other):  return _Quantity(self.mag / (self._m(other) or 1))
    def __rtruediv__(self, other): return _Quantity((self._m(other) or 1) / (self.mag or 1))
    def __add__(self, other):      return _Quantity(self.mag + self._m(other))
    def __radd__(self, other):     return _Quantity(self._m(other) + self.mag)
    def __sub__(self, other):      return _Quantity(self.mag - self._m(other))
    def __rsub__(self, other):     return _Quantity(self._m(other) - self.mag)
    def __pow__(self, exp):        return _Quantity(self.mag) # ignore exponent for docs
    # common helpers
    def value_in_unit(self, u):    return self.mag
    def __repr__(self):            return f"<Quantity ~{self.mag}>"
    __array_priority__ = 10000
    def __array_ufunc__(self, *a, **k): return self

class _Unit:
    """Docs stub for unit constants; behaves like a no-op multiplier."""
    def __mul__(self, other): return _Unit() if not isinstance(other, Real) else _Quantity(other)
    __rmul__ = __mul__
    def __truediv__(self, other): return _Unit()
    def __rtruediv__(self, other): return _Unit()
    def __pow__(self, exp): return _Unit()
    def __repr__(self): return '<unit>'

for name in [
    'angstroms',
    'dalton', 'daltons',
    'nanometer', 'nanometers',
    'picosecond', 'picoseconds',
    'kelvin', 'kilojoule_per_mole',
    'kilojoules_per_mole',
    'kilocalorie_per_mole',
    'kilocalories_per_mole'
]:
    setattr(unit, name, _Unit())

# Core classes referenced at import time
class System: ...
class Context: ...
class Integrator: ...
class Force: ...
class NonbondedForce(Force): ...
class Platform: ...

for attr, obj in {
    'System': System, 'Context': Context, 
    'Integrator': Integrator, 'Force': Force, 
    'NonbondedForce': NonbondedForce, 'Platform': Platform, 
}.items():
    setattr(omm, attr, obj)

# app-layer classes
class Simulation:
    def __init__(self, *args, **kwargs): ...
    def step(self, n): ...
class Topology: ...
class Modeller: ...
class PDBFile:
    def __init__(self, *args, **kwargs): ...
class AmberPrmtopFile:
    def __init__(self, *args, **kwargs): ...
class AmberInpcrdFile:
    def __init__(self, *args, **kwargs): ...

for attr, obj in {
    'Simulation': Simulation, 'Topology': Topology,
    'Modeller': Modeller, 'PDBFile': PDBFile,
    'AmberPrmtopFile': AmberPrmtopFile, 
    'AmberInpcrdFile': AmberInpcrdFile
}.items():
    setattr(app, attr, obj)

singleton = types.ModuleType('openmm.app.internal.singleton')

# register shim modules
sys.modules['openmm'] = omm
sys.modules['openmm.unit'] = unit
sys.modules['openmm.app'] = app
sys.modules['openmm.app.internal'] = app_internal
sys.modules['openmm.app.internal.singleton'] = singleton
# --- end shim ---

# add sphinx extensions and autodoc configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodock_mock_imports = [
    'MDAnalysis',
    'openmm',
    'openbabel',
    'parmed',
    'rdkit',
    'pdbfixer'
]

autodoc_mock_imports = ['MDAnalysis', 'openmm', 'pdbfixer', 'rdkit']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
