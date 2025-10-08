from pathlib import Path
import pytest

from molecular_simulations.build.build_amber import ImplicitSolvent

def test_write_leap_writes_file(tmp_path: Path):
    content = 'source leaprc.protein.ff19SB\nquit\n'
    imp = ImplicitSolvent(path=tmp_path, pdb='dummy.pdb')
    leap_path = imp.write_leap(content)
    assert leap_path.exists()
    assert leap_path.read_text() == content

def test_write_leap_returns_path(tmp_path: Path):
    imp = ImplicitSolvent(path=tmp_path, pdb='dummy.pdb')
    p = imp.write_leap('quit\n')
    assert p.parent == tmp_path
    assert p.suffix in ('.in', '.inp', '.leap', '.txt')
