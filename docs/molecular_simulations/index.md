Module molecular_simulations
============================

Sub-modules
-----------
* molecular_simulations.analysis
* molecular_simulations.build
* molecular_simulations.simulate

Classes
-------

`AuroraSettings(**data: Any)`
:   Compute settings (HPC platform, number of GPUs, etc).
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * molecular_simulations.BaseComputeSettings
    * abc.ABC
    * molecular_simulations.BaseSettings
    * pydantic.main.BaseModel

    ### Class variables

    `account: str`
    :

    `cpus_per_node: int`
    :

    `model_config`
    :

    `num_nodes: int`
    :

    `queue: str`
    :

    `retries: int`
    :

    `scheduler_options: str`
    :

    `strategy: str`
    :

    `walltime: str`
    :

    `worker_init: str`
    :

    ### Methods

    `config_factory(self, run_dir: str | pathlib.Path) ‑> parsl.config.Config`
    :   Create a Parsl configuration for running on Aurora.

`BaseComputeSettings(**data: Any)`
:   Compute settings (HPC platform, number of GPUs, etc).
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * abc.ABC
    * molecular_simulations.BaseSettings
    * pydantic.main.BaseModel

    ### Descendants

    * molecular_simulations.AuroraSettings
    * molecular_simulations.PolarisSettings

    ### Class variables

    `model_config`
    :

    ### Methods

    `config_factory(self, run_dir: str | pathlib.Path) ‑> parsl.config.Config`
    :   Create new Parsl configuration.

`BaseSettings(**data: Any)`
:   !!! abstract "Usage Documentation"
        [Models](../concepts/models.md)
    
    A base class for creating Pydantic models.
    
    Attributes:
        __class_vars__: The names of the class variables defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The synthesized `__init__` [`Signature`][inspect.Signature] of the model.
    
        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The core schema of the model.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a [`RootModel`][pydantic.root_model.RootModel].
        __pydantic_serializer__: The `pydantic-core` `SchemaSerializer` used to dump instances of the model.
        __pydantic_validator__: The `pydantic-core` `SchemaValidator` used to validate instances of the model.
    
        __pydantic_fields__: A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.
        __pydantic_computed_fields__: A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.
    
        __pydantic_extra__: A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
            is set to `'allow'`.
        __pydantic_fields_set__: The names of fields explicitly set during instantiation.
        __pydantic_private__: Values of private attributes set on the model instance.
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Descendants

    * molecular_simulations.BaseComputeSettings

    ### Class variables

    `model_config`
    :

    ### Static methods

    `from_yaml(filename: str | pathlib.Path) ‑> ~_T`
    :

    ### Methods

    `dump_yaml(self, filename: str | pathlib.Path) ‑> None`
    :

`PolarisSettings(**data: Any)`
:   Compute settings (HPC platform, number of GPUs, etc).
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `self` is explicitly positional-only to allow `self` as a field name.

    ### Ancestors (in MRO)

    * molecular_simulations.BaseComputeSettings
    * abc.ABC
    * molecular_simulations.BaseSettings
    * pydantic.main.BaseModel

    ### Class variables

    `account: str`
    :

    `cpus_per_node: int`
    :

    `model_config`
    :

    `num_nodes: int`
    :

    `queue: str`
    :

    `scheduler_options: str`
    :

    `strategy: str`
    :

    `walltime: str`
    :

    `worker_init: str`
    :

    ### Methods

    `config_factory(self, run_dir: str | pathlib.Path) ‑> parsl.config.Config`
    :   Create a configuration suitable for running all tasks on single nodes of Polaris
        We will launch 4 workers per node, each pinned to a different GPU
        Args:
            num_nodes: Number of nodes to use for the MPI parallel tasks
            user_options: Options for which account to use, location of environment files, etc
            run_dir: Directory in which to store Parsl run files. Default: `runinfo`