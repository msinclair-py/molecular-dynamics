"""
"""
import dill as pickle
from openmm import Context, NonbondedForce, GBSAOBCForce
from openmm.app import element, Modeller, Simulation
from openmm.app.forcefield import NonbondedGenerator
from openmm.app.internal import compiled
from openmm.unit import nanometers, kelvin, elementary_charge, is_quantity, MOLAR_GAS_CONSTANT_R
from openmm.unit import sum as unitsum
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
import numpy as np

class ResidueState(object):
    def __init__(self, residueIndex, atomIndices, particleParameters, exceptionParameters, numHydrogens):
        self.residueIndex = residueIndex
        self.atomIndices = atomIndices
        self.particleParameters = particleParameters
        self.exceptionParameters = exceptionParameters
        self.numHydrogens = numHydrogens

class ResidueTitration(object):
    def __init__(self, variants, referenceEnergies):
        self.variants = variants
        self.referenceEnergies = referenceEnergies
        self.explicitStates = []
        self.implicitStates = []
        self.explicitHydrogenIndices = []
        self.protonatedIndex = -1
        self.currentIndex = -1

class ConstantPH(object):
    """
    Construct a ConstantPH object that can be used to run a simulation at constant pH.

    Parameters
    ----------
    topology: openmm.app.Topology
        This describes to model to simulate.  If a residue can exist in multiple protonation states, this Topology may
        use any one of them.  The alternate versions will be constructed by calling `Modeller.addHydrogens()`.
    positions: list
        The initial positions of the atoms
    pH: float or list
        The pH to perform the simulation at.  If this is a single number, the simulation will be at a single fixed pH.
        If it is a list of numbers, simulated tempering will be used to explore a range of pH values.  Each time a
        Monte Carlo move is attempt, the simulation might also transition to a different pH.
    explicitForceField: openmm.app.ForceField
        The force field to use for parameterizing the system used in the simulation.  This should correspond to
        explicit solvent.
    implicitForceField: openmm.app.ForceField
        The force field to use for parameterizing the system used in evaluating Monte Carlo moves.  This should
        correspond to implicit solvent.
    residueVariants: dict
        This should contain one entry for every titratable residue.  Each key should be the index of a residue in the
        Topology.  The corresponding value should be a list of variants that can be created by passing them to
        `Modeller.addHydrogens()`.  Most often they will simply be strings, for example `{1: ['CYS', 'CYX']}`, but
        they can also be detailed descriptions of the exact hydrogens to add.  See the documentation on `addHydrogens()`
        for details.
    referenceEnergies: dict
        The reference energies of the titratable residues in a similar format to `residueVariants`.  Each key should be
        the index of a residue in the Topology.  The corresponding value should be a list of energies in the same order
        as `residueVariants`.
    relaxationSteps: int
        The number of integration steps to perform to relax solvent after a Monte Carlo move is accepted.
    explicitArgs: dict
        Any arguments to pass to `ForceField.createSystem()` when creating the explicit solvent system.
    implicitArgs: dict
        Any arguments to pass to `ForceField.createSystem()` when creating the implicit solvent system.
    integrator: openmm.Integrator
        The integrator to use for the simulation.
    relaxationIntegrator: openmm.Integrator
        The integrator to use for relaxing solvent after an accepted Monte Carlo move.
    weights: list, optional
        The weight factor to use for each pH in the simulated tempering algorithm.  This may be None, in which case
        weights are determined automatically with the Wang-Landau algorithm.  In that case, it takes some time for the
        weights to converge.  Data from that initial period should be discarded; until the weights have converged, the
        simulation does not follow the correct distribution.
    platform: openmm.Platform, optional
        The Platform to use for running the simulation.  If this is None, a Platform is selected automatically.
    properties: dict, optional
        Platform-specific properties to pass to the Context's constructor.
    """
    def __init__(self, topology, positions, pH, explicitForceField, implicitForceField, residueVariants, referenceEnergies,
                 relaxationSteps, explicitArgs, implicitArgs, integrator, relaxationIntegrator, weights=None, platform=None, properties=None):
        if not isinstance(pH, Sequence):
            pH = [pH]
        self.setPH(pH, weights)
        self.currentPHIndex = 0
        self._explicitArgs = explicitArgs
        self._implicitArgs = implicitArgs
        self.relaxationSteps = relaxationSteps
        self.titrations = {}
        for resIndex, variants in residueVariants.items():
            energies = list(referenceEnergies[resIndex])
            self.titrations[resIndex] = ResidueTitration(variants, energies)
        implicitToExplicitResidueMap = []
        explicitToImplicitResidueMap = {}
        solventResidues = []

        # Build the implicit solvent topology by removing water and ions.

        ionElements = (element.cesium, element.potassium, element.lithium, element.sodium, element.rubidium,
                       element.chlorine, element.bromine, element.fluorine, element.iodine)
        for residue in topology.residues():
            if residue.name == 'HOH' or (len(residue) == 1 and next(residue.atoms()).element in ionElements):
                solventResidues.append(residue)
            else:
                implicitToExplicitResidueMap.append(residue.index-len(solventResidues))
        for i, j in enumerate(implicitToExplicitResidueMap):
            explicitToImplicitResidueMap[j] = i
        modeller = Modeller(topology, positions)
        modeller.delete(solventResidues)
        implicitTopology = modeller.topology
        implicitPositions = modeller.positions

        # Loop over variants to construct a ResidueState for every variant of every titratable residue.

        variantIndex = 0
        finished = False
        explicitVariants = [None]*topology.getNumResidues()
        implicitVariants = [None]*implicitTopology.getNumResidues()
        while not finished:
            finished = True

            # Build the explicit solvent states.

            for resIndex, variants in residueVariants.items():
                if variantIndex < len(variants):
                    finished = False
                    explicitVariants[resIndex] = variants[variantIndex]

            explicitStates = self._findResidueStates(topology, positions, explicitForceField, explicitVariants, explicitArgs)

            # Build the implicit solvent states.

            for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
                if explicitIndex in residueVariants:
                    variants = residueVariants[explicitIndex]
                    if variantIndex < len(variants):
                        implicitVariants[implicitIndex] = variants[variantIndex]
            implicitStates = self._findResidueStates(implicitTopology, implicitPositions, implicitForceField, implicitVariants, implicitArgs)
            assert len(explicitStates) == len(implicitStates)

            # Add them to the ResidueTitration.

            for explicitState, implicitState in zip(explicitStates, implicitStates):
                titration = self.titrations[explicitState.residueIndex]
                if variantIndex < len(titration.variants):
                    titration.explicitStates.append(explicitState)
                    titration.implicitStates.append(implicitState)
            variantIndex += 1

        # Create final versions of the topologies, including the fully protonated versions of all residues.

        for titration in self.titrations.values():
            titration.protonatedIndex = np.argmax([len(state.atomIndices) for state in titration.explicitStates])
        variants = [None]*topology.getNumResidues()
        for resIndex in residueVariants:
            titration = self.titrations[resIndex]
            variants[resIndex] = titration.variants[titration.protonatedIndex]
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=explicitForceField, variants=variants)
        self.explicitTopology = modeller.topology
        explicitPositions = modeller.positions
        variants = [None]*implicitTopology.getNumResidues()
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            if explicitIndex in residueVariants:
                titration = self.titrations[explicitIndex]
                variants[explicitIndex] = titration.variants[titration.protonatedIndex]
        modeller = Modeller(implicitTopology, implicitPositions)
        modeller.addHydrogens(forcefield=implicitForceField, variants=variants)
        self.implicitTopology = modeller.topology
        implicitPositions = modeller.positions
        explicitResidues = list(self.explicitTopology.residues())
        implicitResidues = list(self.implicitTopology.residues())

        # Create systems for them.  Also create a third system that is identical to the explicit one,
        # but freezes non-solvent atoms.

        explicitSystem = explicitForceField.createSystem(self.explicitTopology, **explicitArgs)
        implicitSystem = implicitForceField.createSystem(self.implicitTopology, **implicitArgs)
        relaxationSystem = deepcopy(explicitSystem)
        for residue in self.explicitTopology.residues():
            if residue.name != 'HOH' and (len(residue) > 1 or next(residue.atoms()).element not in ionElements):
                for atom in residue.atoms():
                    relaxationSystem.setParticleMass(atom.index, 0.0)

        # For each ResidueTitration, identify the fully protonated state.  Replace the other states
        # with ones that include all protons, setting the parameters of the missing ones to 0.

        for resIndex, titration in self.titrations.items():
            protonated = titration.protonatedIndex
            titration.currentIndex = protonated
            explicitProtonatedParams = titration.explicitStates[protonated].particleParameters
            implicitProtonatedParams = titration.implicitStates[protonated].particleParameters
            explicitProtonatedExceptionParams = titration.explicitStates[protonated].exceptionParameters
            implicitProtonatedExceptionParams = titration.implicitStates[protonated].exceptionParameters
            explicitAtomIndices = {atom.name: atom.index for atom in explicitResidues[resIndex].atoms()}
            implicitAtomIndices = {atom.name: atom.index for atom in implicitResidues[explicitToImplicitResidueMap[resIndex]].atoms()}
            for i in range(len(titration.explicitStates)):
                if i != protonated:
                    oldExplicit = titration.explicitStates[i]
                    oldImplicit = titration.implicitStates[i]
                    newExplicit = deepcopy(titration.explicitStates[protonated])
                    newImplicit = deepcopy(titration.implicitStates[protonated])
                    newExplicit.numHydrogens = oldExplicit.numHydrogens
                    newImplicit.numHydrogens = oldImplicit.numHydrogens
                    for forceIndex in newExplicit.particleParameters:
                        params = oldExplicit.particleParameters[forceIndex]
                        for atomName in newExplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newExplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newExplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(explicitProtonatedParams[forceIndex][atomName], explicitSystem.getForce(forceIndex))
                                titration.explicitHydrogenIndices.append(explicitAtomIndices[atomName])
                    for forceIndex in newExplicit.exceptionParameters:
                        params = oldExplicit.exceptionParameters[forceIndex]
                        for key in newExplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newExplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newExplicit.exceptionParameters[forceIndex][key] = [0.0]+list(explicitProtonatedExceptionParams[forceIndex][key][1:])
                    for forceIndex in newImplicit.particleParameters:
                        params = oldImplicit.particleParameters[forceIndex]
                        for atomName in newImplicit.particleParameters[forceIndex]:
                            if atomName in params:
                                newImplicit.particleParameters[forceIndex][atomName] = params[atomName]
                            else:
                                newImplicit.particleParameters[forceIndex][atomName] = self._get_zero_parameters(implicitProtonatedParams[forceIndex][atomName], implicitSystem.getForce(forceIndex))
                    for forceIndex in newImplicit.exceptionParameters:
                        params = oldImplicit.exceptionParameters[forceIndex]
                        for key in newImplicit.exceptionParameters[forceIndex]:
                            if key in params:
                                newImplicit.exceptionParameters[forceIndex][key] = params[key]
                            else:
                                newImplicit.exceptionParameters[forceIndex][key] = [0.0]+list(implicitProtonatedExceptionParams[forceIndex][key][1:])
                    titration.explicitStates[i] = newExplicit
                    titration.implicitStates[i] = newImplicit
            for i in range(len(titration.explicitStates)):
                titration.explicitStates[i].atomIndices = explicitAtomIndices
                titration.implicitStates[i].atomIndices = implicitAtomIndices

        # Create contexts or simulations for all the systems.

        self.simulation = Simulation(self.explicitTopology, explicitSystem, deepcopy(integrator), platform, properties)
        platform = self.simulation.context.getPlatform()
        if properties is None:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform)
            self.relaxationContext = Context(relaxationSystem, deepcopy(relaxationIntegrator), platform)
        else:
            self.implicitContext = Context(implicitSystem, deepcopy(integrator), platform, properties)
            self.relaxationContext = Context(relaxationSystem, deepcopy(relaxationIntegrator), platform, properties)
        self.simulation.context.setPositions(explicitPositions)
        self.relaxationContext.setPositions(explicitPositions)
        self.implicitContext.setPositions(implicitPositions)

        # Record the mapping from implicit system atoms to explicit system atoms.  We need this
        # for copying positions.

        implicitAtomIndex = [None]*implicitSystem.getNumParticles()
        for implicitIndex, explicitIndex in enumerate(implicitToExplicitResidueMap):
            explicitRes = explicitResidues[explicitIndex]
            implicitRes = implicitResidues[implicitIndex]
            explicitAtoms = {atom.name: atom.index for atom in explicitRes.atoms()}
            for atom in implicitRes.atoms():
                implicitAtomIndex[atom.index] = explicitAtoms[atom.name]
        self.implicitAtomIndex = np.array(implicitAtomIndex)

        # Record the indices of nonbonded exceptions in each of the contexts.

        self.explicitExceptionIndex = self._findExceptionIndices(explicitSystem, self.explicitTopology)
        self.implicitExceptionIndex = self._findExceptionIndices(implicitSystem, self.implicitTopology)
        self.explicitInterResidue14 = self._findInterResidue14(explicitSystem, self.explicitTopology)
        self.implicitInterResidue14 = self._findInterResidue14(implicitSystem, self.implicitTopology)

        # Record the scale factors for 1-4 Coulomb interactions.

        self.explicit14Scale = self._find14Scale(explicitForceField)
        self.implicit14Scale = self._find14Scale(implicitForceField)

    def setPH(self, pH, weights=None):
        """
        Set the pH to run the simulation at.  See the description of the `pH` and `weights` arguments to the constructor
        for more details.
        """
        self.pH = pH
        if weights is None:
            self._weights = [0.0]*len(pH)
            self._updateWeights = True
            self._weightUpdateFactor = 1.0
            self._histogram = [0]*len(pH)
            self._hasMadeTransition = False
        else:
            self._weights = weights
            self._updateWeights = False

    @property
    def weights(self):
        """
        Get the current values of the weights used in the simulated tempering algorithm.  This has one value for each pH.
        """
        return [x-self._weights[0] for x in self._weights]

    def attemptMCStep(self, temperature):
        """
        Attempt to change the protonation states of all titratable residues.  If simulated tempering is being used, this
        will also attempt to change to a new pH

        Parameters
        ----------
        temperature: float
            the temperature the simulation is being run at
        """
        # Copy the positions to the implicit context.

        state = self.simulation.context.getState(getPositions=True, getParameters=True)
        explicitPositions = state.getPositions(asNumpy=True).value_in_unit(nanometers)
        implicitPositions = explicitPositions[self.implicitAtomIndex]
        self.implicitContext.setPositions(implicitPositions)
        periodicDistance = compiled.periodicDistance(state.getPeriodicBoxVectors().value_in_unit(nanometers))

        # Perform simulated tempering.

        if len(self.pH) > 1:
            self._attemptPHChange()

        # Process the residues in random order.

        anyChange = False
        for resIndex in np.random.permutation(list(self.titrations)):
            titrations = [self.titrations[resIndex]]

            # Select a new state for it.

            stateIndex = [self._selectNewState(titrations[0])]
            if np.random.random() < 0.25:
                # Consider a multisite titration in which two residues change.

                neighbors = self._findNeighbors(resIndex, explicitPositions, periodicDistance)
                if len(neighbors) > 0:
                    i = np.random.choice(neighbors)
                    titrations.append(self.titrations[i])
                    stateIndex.append(self._selectNewState(titrations[-1]))

            # Compute the energy of the implicit solvent system in the current and new states.

            currentEnergy = self.implicitContext.getState(getEnergy=True).getPotentialEnergy()
            for i, t in zip(stateIndex, titrations):
                self._applyStateToContext(t.implicitStates[i], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
            newEnergy = self.implicitContext.getState(getEnergy=True).getPotentialEnergy()

            # Decide whether to accept the new state.

            if not is_quantity(temperature):
                temperature = temperature*kelvin
            kT = (MOLAR_GAS_CONSTANT_R*temperature)
            deltaRefEnergy = unitsum([t.referenceEnergies[i] - t.referenceEnergies[t.currentIndex] for i, t in zip(stateIndex, titrations)])
            deltaN = unitsum([t.implicitStates[i].numHydrogens - t.implicitStates[t.currentIndex].numHydrogens for i, t in zip(stateIndex, titrations)])
            w = (newEnergy-currentEnergy-deltaRefEnergy)/kT + deltaN*np.log(10.0)*self.pH[self.currentPHIndex]
            if w > 0.0 and np.exp(-w) < np.random.random():
                # Restore the previous state.

                for t in titrations:
                    self._applyStateToContext(t.implicitStates[t.currentIndex], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
                continue
            anyChange = True

            # Apply the new state.

            for i, t in zip(stateIndex, titrations):
                t.currentIndex = i
                self._applyStateToContext(t.explicitStates[i], self.simulation.context, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
                self._applyStateToContext(t.explicitStates[i], self.relaxationContext, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)

        # If anything changed, run some dynamics to let the water relax.

        if anyChange:
            self.relaxationContext.setPositions(explicitPositions)
            self.relaxationContext.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            for param in self.relaxationContext.getParameters():
                self.relaxationContext.setParameter(param, state.getParameters()[param])
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            relaxedPositions = self.relaxationContext.getState(getPositions=True).getPositions(asNumpy=True)
            self.simulation.context.setPositions(relaxedPositions)

    def setResidueState(self, residueIndex, stateIndex, relax=False):
        """
        Set a titratable residue to be in a particular state.

        Parameters
        ----------
        residueIndex: int
            The index of the residue to modify
        stateIndex: int
            The index of the state to put it into
        relax: bool
            If True, the solvent is allowed to relax after changing the state by immobilizing the solute and performing
            a short simulation.
        """
        titration = self.titrations[residueIndex]
        self._applyStateToContext(titration.explicitStates[stateIndex], self.simulation.context, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
        self._applyStateToContext(titration.explicitStates[stateIndex], self.relaxationContext, self.explicitExceptionIndex, self.explicitInterResidue14, self.explicit14Scale)
        self._applyStateToContext(titration.implicitStates[stateIndex], self.implicitContext, self.implicitExceptionIndex, self.implicitInterResidue14, self.implicit14Scale)
        titration.currentIndex = stateIndex
        if relax:
            self.relaxationContext.setPositions(self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True))
            self.relaxationContext.getIntegrator().step(self.relaxationSteps)
            self.simulation.context.setPositions(self.relaxationContext.getState(getPositions=True).getPositions(asNumpy=True))

    def _findResidueStates(self, topology, positions, forcefield, variants, ffargs):
        """Given a ForceField and a list of variants for the variable residues, construct ResidueState objects for them."""
        modeller = Modeller(topology, positions)
        modeller.addHydrogens(forcefield=forcefield, variants=variants)
        system = forcefield.createSystem(modeller.topology, **ffargs)
        atoms = list(modeller.topology.atoms())
        residues = list(modeller.topology.residues())
        states = []
        for residue, variant in zip(residues, variants):
            if variant is not None:
                atomIndices = {atom.name: atom.index for atom in residue.atoms()}
                particleParameters = {}
                exceptionParameters = {}
                for i, force in enumerate(system.getForces()):
                    try:
                        particleParameters[i] = {atom.name: force.getParticleParameters(atom.index) for atom in residue.atoms()}
                    except:
                        pass
                    if isinstance(force, NonbondedForce):
                        exceptionParameters[i] = {}
                        for j in range(force.getNumExceptions()):
                            p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(j)
                            atom1 = atoms[p1]
                            atom2 = atoms[p2]
                            if atom1.residue == residue and atom2.residue == residue:
                                exceptionParameters[i][(residue.index, atom1.name, atom2.name)] = (chargeProd, sigma, epsilon)
                numHydrogens = sum(1 for atom in residue.atoms() if atom.element == element.hydrogen)
                states.append(ResidueState(residue.index, atomIndices, particleParameters, exceptionParameters, numHydrogens))
        return states

    def _findExceptionIndices(self, system, topology):
        """Construct a dict whose keys are (residue index, atom 1 name, atom 2 name), and whose values are the indices
        of the corresponding exceptions in the NonbondedForce.  This is needed for mapping exceptions between Topologies
        with different sets of atoms."""
        indices = {}
        atoms = list(topology.atoms())
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue == atom2.residue:
                        indices[(atom1.residue.index, atom1.name, atom2.name)] = i
                        indices[(atom1.residue.index, atom2.name, atom1.name)] = i
        return indices

    def _findInterResidue14(self, system, topology):
        """For each residue, record the indices of all 1-4 exceptions that span that residue and another one."""
        indices = defaultdict(list)
        atoms = list(topology.atoms())
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    atom1 = atoms[p1]
                    atom2 = atoms[p2]
                    if atom1.residue != atom2.residue and chargeProd.value_in_unit(elementary_charge**2) != 0.0:
                        indices[atom1.residue.index].append(i)
                        indices[atom2.residue.index].append(i)
        return indices

    def _find14Scale(self, forcefield):
        """Find the scale factor for 1-4 Coulomb interactions."""
        for generator in forcefield.getGenerators():
            if isinstance(generator, NonbondedGenerator):
                return generator.coulomb14scale
        return 1.0

    def _get_zero_parameters(self, original_parameters, force):
        """Get the per-particle parameter values that should be used to set an atom's charge to 0."""
        p = list(original_parameters)
        if isinstance(force, NonbondedForce) or isinstance(force, GBSAOBCForce):
            p[0] = 0.0
        else:
            for i in range(force.getNumPerParticleParameters()):
                if force.getPerParticleParameterName(i) == 'charge':
                    p[i] = 0.0
        return tuple(p)

    def _applyStateToContext(self, state, context, exceptionIndex, interResidue14, coulomb14Scale):
        """Given a ResidueState, update parameters in a Context to match that state."""
        for forceIndex, params in state.particleParameters.items():
            force = context.getSystem().getForce(forceIndex)
            for atomName, atomParams in params.items():
                atomIndex = state.atomIndices[atomName]
                try:
                    # Custom forces take the parameters as a single tuple.
                    force.setParticleParameters(atomIndex, atomParams)
                except:
                    # Standard forces take them as separate arguments.
                    force.setParticleParameters(atomIndex, *atomParams)
            if isinstance(force, NonbondedForce):
                for key, exceptionParams in state.exceptionParameters[forceIndex].items():
                    p = force.getExceptionParameters(exceptionIndex[key])
                    force.setExceptionParameters(exceptionIndex[key], p[0], p[1], *exceptionParams)
                for index in interResidue14[state.residueIndex]:
                    p1, p2, _, sigma, epsilon = force.getExceptionParameters(index)
                    q1, _, _ = force.getParticleParameters(p1)
                    q2, _, _ = force.getParticleParameters(p2)
                    force.setExceptionParameters(index, p1, p2, coulomb14Scale*q1*q2, sigma, epsilon)
            context.reinitialize(preserveState=True)
            force.updateParametersInContext(context)

    def _selectNewState(self, titration):
        """Randomly choose a new state for a ResidueTitration."""
        numStates = len(titration.implicitStates)
        if numStates == 2:
            return 1-titration.currentIndex
        stateIndex = titration.currentIndex
        while stateIndex == titration.currentIndex:
            stateIndex = np.random.randint(numStates)
        return stateIndex

    def _findNeighbors(self, resIndex, explicitPositions, periodicDistance):
        """Find other titratable residues that are very close to a specified residue.  This is used for
        multisite titrations."""
        neighbors = []
        titration1 = self.titrations[resIndex]
        for resIndex2 in self.titrations:
            if resIndex2 > resIndex:
                titration2 = self.titrations[resIndex2]
                isNeighbor = False
                for i in titration1.explicitHydrogenIndices:
                    for j in titration2.explicitHydrogenIndices:
                        if periodicDistance(explicitPositions[i], explicitPositions[j]) < 0.2:
                            isNeighbor = True
                if isNeighbor:
                    neighbors.append(resIndex2)
        return neighbors

    def _attemptPHChange(self):
        """Attempt to change to a different pH."""
        # Compute the probability for each pH.  This is done in log space to avoid overflow.

        hydrogens = sum(t.explicitStates[t.currentIndex].numHydrogens for t in self.titrations.values())
        logProbability = [(self._weights[i]-hydrogens*np.log(10.0)*self.pH[i]) for i in range(len(self._weights))]
        maxLogProb = max(logProbability)
        offset = maxLogProb + np.log(sum(np.exp(x-maxLogProb) for x in logProbability))
        probability = [np.exp(x-offset) for x in logProbability]
        r = np.random.random_sample()
        for j in range(len(probability)):
            if r < probability[j]:
                if j != self.currentPHIndex:
                    self._hasMadeTransition = True
                self.currentPHIndex = j
                if self._updateWeights:
                    # Update the weight factors.

                    self._weights[j] -= self._weightUpdateFactor
                    self._histogram[j] += 1
                    minCounts = min(self._histogram)
                    if minCounts > 20 and minCounts >= 0.2*sum(self._histogram)/len(self._histogram):
                        # Reduce the weight update factor and reset the histogram.

                        self._weightUpdateFactor *= 0.5
                        self._histogram = [0]*len(self.pH)
                        self._weights = [x-self._weights[0] for x in self._weights]
                    elif not self._hasMadeTransition and probability[self.currentPHIndex] > 0.99 and self._weightUpdateFactor < 1024.0:
                        # Rapidly increase the weight update factor at the start of the simulation to find
                        # a reasonable starting value.

                        self._weightUpdateFactor *= 2.0
                        self._histogram = [0]*len(self.pH)
                return
            r -= probability[j]
