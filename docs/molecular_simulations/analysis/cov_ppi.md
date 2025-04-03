Module molecular_simulations.analysis.cov_ppi
=============================================

Classes
-------

`PPInteractions(top: str | pathlib.Path, traj: str | pathlib.Path, out: str | pathlib.Path, sel1: str = 'chainID A', sel2: str = 'chainID B', cov_cutoff: Tuple[float] = (11.0, 13.0), sb_cutoff: float = 6.0, hbond_cutoff: float = 3.5, hbond_angle: float = 30.0, hydrophobic_cutoff: float = 8.0, plot: bool = True)`
:   Code herein adapted from: 
        https://www.biorxiv.org/content/10.1101/2025.03.24.644990v1.full.pdf
    Takes an input topology file and trajectory file, and highlights relevant
    interactions between two selections. To this end we first compute the 
    covariance matrix between the two selections, filter out all interactions
    which occur too far apart (11Å for positive covariance, 13Å for negative
    covariance), and examines each based on a variety of distance and angle
    cutoffs defined in the literature.

    ### Methods

    `analyze_hbond(self, res1: MDAnalysis.core.groups.AtomGroup, res2: MDAnalysis.core.groups.AtomGroup) ‑> float`
    :   Identifies all potential donor/acceptor atoms between two
        residues. Culls this list based on distance array across simulation
        and then evaluates each pair over the trajectory utilizing a
        distance and angle cutoff.

    `analyze_hydrophobic(self, res1: MDAnalysis.core.groups.AtomGroup, res2: MDAnalysis.core.groups.AtomGroup) ‑> float`
    :

    `analyze_saltbridge(self, res1: MDAnalysis.core.groups.AtomGroup, res2: MDAnalysis.core.groups.AtomGroup) ‑> float`
    :   Uses a simple distance cutoff to highlight the occupancy of 
        saltbridge between two residues. Returns the fraction of
        simulation time spent engaged in saltbridge.

    `compute_interactions(self, res1: int, res2: int) ‑> Dict[str, Dict[str, float]]`
    :   Ingests two resIDs, generates MDAnalysis AtomGroups for each, identifies
        relevant non-bonded interactions (HBonds, saltbridge, hydrophobic) and
        computes each. Returns a Dict containing the proportion of simulation time
        that each interaction is engaged.

    `evaluate_hbond(sel, donor: MDAnalysis.core.groups.AtomGroup, acceptor: MDAnalysis.core.groups.AtomGroup) ‑> int`
    :   Evaluates whether there is a defined hydrogen bond between any
        donor and acceptor atoms in a given frame. Must pass a distance
        cutoff as well as an angle cutoff. Returns early when a legal
        HBond is detected.

    `get_covariance(self) ‑> numpy.ndarray`
    :   Loop over all C-alpha atoms and compute the positional
        covariance using the functional form:
            C = <(R1 - <R1>)(R2 - <R2>)T>
        where each element corresponds to the ensemble average movement
            C_ij = <deltaR_i * deltaR_j>
        with the magnitude being the strength of correlation and the sign
        corresponding to positive and negative correlation respectively.

    `identify_interaction_type(self, res1: str, res2: str) ‑> List[Callable]`
    :   Identifies what analyses to compute for a given pair of protein
        residues (i.e. hydrophobic interactions, hydrogen bonds, saltbridges).

    `interpret_covariance(self, cov_mat: numpy.ndarray) ‑> Tuple[Tuple[int, int]]`
    :   Identify pairs of residues with positive or negative correlations.
        Returns a tuple comprised of pairs for each.

    `make_plot(self, data: pandas.core.frame.DataFrame, column: str, name: str | pathlib.Path, fs: int = 15) ‑> None`
    :

    `parse_results(self, results: Dict[str, Dict[str, float]]) ‑> pandas.core.frame.DataFrame`
    :   Prepares results for plotting. Removes any entries which are
        all 0. and returns as a pandas DataFrame for easier plotting.

    `plot_results(self, results: Dict[str, Dict[str, float]]) ‑> None`
    :

    `res_map(self, ag1: MDAnalysis.core.groups.AtomGroup, ag2: MDAnalysis.core.groups.AtomGroup) ‑> None`
    :   Map covariance matrix indices to AtomGroup resIDs so that we are
        examining the correct pairs of residues.

    `run(self) ‑> None`
    :   Main function that runs the workflow. Obtains a covariance matrix,
        screens for close interactions, evaluates each pairwise interaction
        for each amino acid and report the contact probability of each.

    `save(self, results: Dict[str, Dict[str, float]]) ‑> None`
    :   Save results as a json file.

    `survey_donors_acceptors(self, res1: MDAnalysis.core.groups.AtomGroup, res2: MDAnalysis.core.groups.AtomGroup) ‑> Tuple[MDAnalysis.core.groups.AtomGroup]`
    :   First pass distance threshhold to identify potential Hydrogen bonds.
        Should be followed by querying HBond angles but this serves to reduce
        our search space and time complexity. Only returns donors/acceptors which
        are within the distance cutoff in at least a single frame.