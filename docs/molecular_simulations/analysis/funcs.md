Module molecular_simulations.analysis.funcs
===========================================

Classes
-------

`CoMDist(target: MDAnalysis.core.groups.AtomGroup, binder: MDAnalysis.core.groups.AtomGroup)`
:   Base class for defining multi-frame analysis
    
    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.
    
    To define a new Analysis, :class:`AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the
    :class:`MDAnalysis.analysis.results.Results` container.
    
    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    verbose : bool, optional
        Turn on more logging and debugging
    
    Attributes
    ----------
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    results: :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    
    
    Example
    -------
    .. code-block:: python
    
       from MDAnalysis.analysis.base import AnalysisBase
    
       class NewAnalysis(AnalysisBase):
           def __init__(self, atomgroup, parameter, **kwargs):
               super(NewAnalysis, self).__init__(atomgroup.universe.trajectory,
                                                 **kwargs)
               self._parameter = parameter
               self._ag = atomgroup
    
           def _prepare(self):
               # OPTIONAL
               # Called before iteration on the trajectory has begun.
               # Data structures can be set up at this time
               self.results.example_result = []
    
           def _single_frame(self):
               # REQUIRED
               # Called after the trajectory is moved onto each new frame.
               # store an example_result of `some_function` for a single frame
               self.results.example_result.append(some_function(self._ag,
                                                                self._parameter))
    
           def _conclude(self):
               # OPTIONAL
               # Called once iteration on the trajectory is finished.
               # Apply normalisation and averaging to results here.
               self.results.example_result = np.asarray(self.example_result)
               self.results.example_result /=  np.sum(self.result)
    
    Afterwards the new analysis can be run like this
    
    .. code-block:: python
    
       import MDAnalysis as mda
       from MDAnalysisTests.datafiles import PSF, DCD
    
       u = mda.Universe(PSF, DCD)
    
       na = NewAnalysis(u.select_atoms('name CA'), 35)
       na.run(start=10, stop=20)
       print(na.results.example_result)
       # results can also be accessed by key
       print(na.results["example_result"])
    
    
    .. versionchanged:: 1.0.0
        Support for setting `start`, `stop`, and `step` has been removed. These
        should now be directly passed to :meth:`AnalysisBase.run`.
    
    .. versionchanged:: 2.0.0
        Added :attr:`results`
    
    .. versionchanged:: 2.8.0
        Added ability to run analysis in parallel using either a
        built-in backend (`multiprocessing` or `dask`) or a custom
        `backends.BackendBase` instance with an implemented `apply` method
        that is used to run the computations.

    ### Ancestors (in MRO)

    * MDAnalysis.analysis.base.AnalysisBase

`ContactFrequency(binder, target, cutoff=5.0)`
:   Base class for defining multi-frame analysis
    
    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.
    
    To define a new Analysis, :class:`AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the
    :class:`MDAnalysis.analysis.results.Results` container.
    
    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    verbose : bool, optional
        Turn on more logging and debugging
    
    Attributes
    ----------
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    results: :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    
    
    Example
    -------
    .. code-block:: python
    
       from MDAnalysis.analysis.base import AnalysisBase
    
       class NewAnalysis(AnalysisBase):
           def __init__(self, atomgroup, parameter, **kwargs):
               super(NewAnalysis, self).__init__(atomgroup.universe.trajectory,
                                                 **kwargs)
               self._parameter = parameter
               self._ag = atomgroup
    
           def _prepare(self):
               # OPTIONAL
               # Called before iteration on the trajectory has begun.
               # Data structures can be set up at this time
               self.results.example_result = []
    
           def _single_frame(self):
               # REQUIRED
               # Called after the trajectory is moved onto each new frame.
               # store an example_result of `some_function` for a single frame
               self.results.example_result.append(some_function(self._ag,
                                                                self._parameter))
    
           def _conclude(self):
               # OPTIONAL
               # Called once iteration on the trajectory is finished.
               # Apply normalisation and averaging to results here.
               self.results.example_result = np.asarray(self.example_result)
               self.results.example_result /=  np.sum(self.result)
    
    Afterwards the new analysis can be run like this
    
    .. code-block:: python
    
       import MDAnalysis as mda
       from MDAnalysisTests.datafiles import PSF, DCD
    
       u = mda.Universe(PSF, DCD)
    
       na = NewAnalysis(u.select_atoms('name CA'), 35)
       na.run(start=10, stop=20)
       print(na.results.example_result)
       # results can also be accessed by key
       print(na.results["example_result"])
    
    
    .. versionchanged:: 1.0.0
        Support for setting `start`, `stop`, and `step` has been removed. These
        should now be directly passed to :meth:`AnalysisBase.run`.
    
    .. versionchanged:: 2.0.0
        Added :attr:`results`
    
    .. versionchanged:: 2.8.0
        Added ability to run analysis in parallel using either a
        built-in backend (`multiprocessing` or `dask`) or a custom
        `backends.BackendBase` instance with an implemented `apply` method
        that is used to run the computations.

    ### Ancestors (in MRO)

    * MDAnalysis.analysis.base.AnalysisBase

`DeltaCOM(sel: MDAnalysis.core.groups.AtomGroup, sliding_window: int = 1, residence: bool = True, residence_distance_cutoff: float = 0.5, min_residence: int = 5)`
:   Computes the change in CoM in a sliding window, saving the delta distances into
    an array. Then loops over the array to determine which if any resident binding
    events have occurred, how long they occurred for and in what order they occurred.

    ### Ancestors (in MRO)

    * MDAnalysis.analysis.base.AnalysisBase

    ### Methods

    `distance(self, frame: int)`
    :   Euclidean distance is just L2 norm, use numpy.linalg.norm to
        compute this efficiently, or if its the first frame and we throw
        an IndexError return a NaN.

`RadiusofGyration(atomgroup)`
:   Base class for defining multi-frame analysis
    
    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.
    
    To define a new Analysis, :class:`AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the
    :class:`MDAnalysis.analysis.results.Results` container.
    
    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    verbose : bool, optional
        Turn on more logging and debugging
    
    Attributes
    ----------
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    results: :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    
    
    Example
    -------
    .. code-block:: python
    
       from MDAnalysis.analysis.base import AnalysisBase
    
       class NewAnalysis(AnalysisBase):
           def __init__(self, atomgroup, parameter, **kwargs):
               super(NewAnalysis, self).__init__(atomgroup.universe.trajectory,
                                                 **kwargs)
               self._parameter = parameter
               self._ag = atomgroup
    
           def _prepare(self):
               # OPTIONAL
               # Called before iteration on the trajectory has begun.
               # Data structures can be set up at this time
               self.results.example_result = []
    
           def _single_frame(self):
               # REQUIRED
               # Called after the trajectory is moved onto each new frame.
               # store an example_result of `some_function` for a single frame
               self.results.example_result.append(some_function(self._ag,
                                                                self._parameter))
    
           def _conclude(self):
               # OPTIONAL
               # Called once iteration on the trajectory is finished.
               # Apply normalisation and averaging to results here.
               self.results.example_result = np.asarray(self.example_result)
               self.results.example_result /=  np.sum(self.result)
    
    Afterwards the new analysis can be run like this
    
    .. code-block:: python
    
       import MDAnalysis as mda
       from MDAnalysisTests.datafiles import PSF, DCD
    
       u = mda.Universe(PSF, DCD)
    
       na = NewAnalysis(u.select_atoms('name CA'), 35)
       na.run(start=10, stop=20)
       print(na.results.example_result)
       # results can also be accessed by key
       print(na.results["example_result"])
    
    
    .. versionchanged:: 1.0.0
        Support for setting `start`, `stop`, and `step` has been removed. These
        should now be directly passed to :meth:`AnalysisBase.run`.
    
    .. versionchanged:: 2.0.0
        Added :attr:`results`
    
    .. versionchanged:: 2.8.0
        Added ability to run analysis in parallel using either a
        built-in backend (`multiprocessing` or `dask`) or a custom
        `backends.BackendBase` instance with an implemented `apply` method
        that is used to run the computations.

    ### Ancestors (in MRO)

    * MDAnalysis.analysis.base.AnalysisBase

    ### Static methods

    `radgyr(atomgroup, masses, total_mass=None)`
    :