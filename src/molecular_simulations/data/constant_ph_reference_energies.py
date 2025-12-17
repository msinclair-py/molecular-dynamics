def get_ref_energies(ff: str='amber19'):
    match ff.lower():
        case 'amber19':
            ref_energies = {
                'CYS': [0., -322.8469307185402],
                'ASP': [0., -126.57035209132911],
                'GLU': [0., -121.02371348056394],
                'LYS': [0., -87.04237154061295],
                'HIS': [0., -97.76840073921795, -92.98848479789538],
            }
        case _:
            raise ValueError(f'Forcefield {ff} not yet computed!')

    
    return ref_energies
