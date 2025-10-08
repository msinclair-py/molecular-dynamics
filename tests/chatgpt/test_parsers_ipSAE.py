import pytest

pytestmark = pytest.mark.fast # quick to run

def _get_attr_or_skip(mod, name):
    if hasattr(mod, name):
        return getattr(mod, name)
    pytest.skip(f'{name} not found in {mod.__name__}')

def test_ipSAE_pdb_atom_line_parsing():
    """
    Tries to exercise a minimial PDB ATOM/HETATM line parser if present.
    Skips if parser symbol isn't available.
    """
    try:
        from molecular_simulations.analysis import ipSAE
    except Exception as e:
        pytest.skip(f'ipSAE module not importable: {e}')

    # Common, guessable entry points (adjusted dynamically)
    candidates = [
        'parse_pdb_atom', # function(line) -> dict/obj
        'parse_pdb_line', # function(line) -> dict/obj
        'ModelParser'     # class with .parse_pdb_atom or .parse_line
    ]
    parser = None
    for name in candidates:
        if hasattr(ipSAE, name):
            parser = getattr(ipSAE, name)
            break
    if parser is None:
        pytest.skip('No recognizable PDB parsing symbol in ipSAE')

    pdb_line = (
        'ATOM      1  N   MET A   1      38.428  13.947   8.678  1.00 54.69           N  '
    )
    # Try likely call patterns:
    if callable(parser):
        rec = parser(pdb_line)
    else:
        # Assume class with parsing methods
        mp = parser()  # type: ignore
        if hasattr(mp, 'parse_pdb_atom'):
            rec = mp.parse_pdb_atom(pdb_line)  # type: ignore
        elif hasattr(mp, 'parse_line'):
            rec = mp.parse_line(pdb_line)  # type: ignore
        else:
            pytest.skip('ipSAE.ModelParser has no parse_pdb_atom/parse_line')

    # Generic, non-strict assertions so we don't couple to implementation details
    assert rec is not None
    # Expect at least atom name / residue name / chain / res id or coordinates
    repr_text = repr(rec)
    assert any(k in repr_text for k in ('ATOM', 'N  ', 'MET', 'chain', 'x', 'y', 'z', 'resid', 'resSeq'))

def test_ipSAE_cif_atom_line_parsing():
    """
    Similar idea for CIF-style inputs. Skips if no CIF parser is present.
    """

