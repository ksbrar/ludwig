from abc import ABC
from dataclasses import field
from typing import Callable, Dict, Optional, Tuple, Union

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.utils.registry import Registry

# ----------------------------------------------------------------------------------------------------------------------
# To prevent direct dependency on ray import, the following static key stores are duplicated:

# from ray.tune.schedulers import SCHEDULER_IMPORT
# https://github.com/ray-project/ray/blob/137a1b12c3b31a3622fa5f721a05a64e9b559b05/python/ray/tune/schedulers/__init__.py#L28

# from ray.tune.result import DEFAULT_RESULT_KEYS
# Taken from https://github.com/ray-project/ray/blob/137a1b12c3b31a3622fa5f721a05a64e9b559b05/python/ray/tune/result.py
TRAINING_ITERATION = "training_iteration"
TIME_TOTAL_S = "time_total_s"
TIMESTEPS_TOTAL = "timesteps_total"
MEAN_ACCURACY = "mean_accuracy"
MEAN_LOSS = "mean_loss"
DEFAULT_RESULT_KEYS = (TRAINING_ITERATION, TIME_TOTAL_S, TIMESTEPS_TOTAL, MEAN_ACCURACY, MEAN_LOSS)

# from ray.tune.result import DEFAULT_METRIC
RAY_TUNE_DESULT_DEFAULT_METRIC = "_metric"
# ----------------------------------------------------------------------------------------------------------------------

scheduler_config_registry = Registry()


def register_scheduler_config(name: str):
    def wrap(scheduler_config: BaseSchedulerConfig):
        scheduler_config_registry[name] = scheduler_config
        return scheduler_config

    return wrap


@dataclass
class BaseSchedulerConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for schedulers.

    Not meant to be used directly.
    """

    type: str
    """Name corresponding to a scheduler in `ludwig.schema.hyperopt.scheduler.scheduler_registry`.
       Technically mutable, but attempting to load a derived optimizer with `type` set to a mismatched value will
       result in a `ValidationError`."""


@register_scheduler_config("async_hyperband")
@register_scheduler_config("asynchyperband")
@dataclass
class AsyncHyperbandSchedulerConfig(BaseSchedulerConfig):
    """Asynchronous hyperband (ASHA) scheduler settings."""

    type: str = schema_utils.StringOptions(options=["async_hyperband"], default="async_hyperband", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TRAINING_ITERATION,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    max_t: int = schema_utils.PositiveInteger(
        default=100,
        description=(
            "max time units per trial. Trials will be stopped after max_t time units (determined by time_attr) have "
            "passed."
        ),
    )

    grace_period: int = schema_utils.PositiveInteger(
        default=1,
        description=(
            "Only stop trials at least this old in time. The units are the same as the attribute named by time_attr."
        ),
    )

    reduction_factor: int = schema_utils.NonNegativeFloat(
        default=4, description=("Used to set halving rate and amount. This is simply a unit-less scalar.")
    )


@register_scheduler_config("hyperband")
@dataclass
class HyperbandSchedulerConfig(BaseSchedulerConfig):
    """Standard hyperband scheduler settings."""

    type: str = schema_utils.StringOptions(options=["hyperband"], default="hyperband", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TRAINING_ITERATION,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    max_t: int = schema_utils.PositiveInteger(
        default=81,
        description=(
            "max time units per trial. Trials will be stopped after max_t time units (determined by time_attr) have "
            "passed."
        ),
    )

    reduction_factor: int = schema_utils.NonNegativeFloat(
        default=3, description=("Used to set halving rate and amount. This is simply a unit-less scalar.")
    )

    stop_last_trials: bool = schema_utils.Boolean(
        default=True, description=("Whether to terminate the trials after reaching max_t. Defaults to True.")
    )


@register_scheduler_config("median_stopping_rule")
@register_scheduler_config("medianstoppingrule")
@dataclass
class MedianStoppingRuleSchedulerConfig(BaseSchedulerConfig):
    """Median Stopping Rule scheduler settings."""

    type: str = schema_utils.StringOptions(
        options=["median_stopping_rule"], default="median_stopping_rule", allow_none=False
    )

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TIME_TOTAL_S,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    grace_period: float = schema_utils.NonNegativeFloat(
        default=60.0,
        description=(
            "Only stop trials at least this old in time. The mean will only be computed from this time onwards. The "
            "units are the same as the attribute named by time_attr."
        ),
    )

    min_samples_required: int = schema_utils.PositiveInteger(
        default=3, description=("Minimum number of trials to compute median over.")
    )

    min_time_slice: int = schema_utils.NonNegativeInteger(
        default=0,
        description=(
            "Each trial runs at least this long before yielding (assuming it isn’t stopped). Note: trials ONLY yield if"
            " there are not enough samples to evaluate performance for the current result AND there are other trials "
            "waiting to run. The units are the same as the attribute named by time_attr."
        ),
    )

    hard_stop: bool = schema_utils.Boolean(
        default=True,
        description=(
            "If False, pauses trials instead of stopping them. When all other trials are complete, paused trials will "
            "be resumed and allowed to run FIFO."
        ),
    )


@register_scheduler_config("pbt")
@dataclass
class PopulationBasedTrainingSchedulerConfig(BaseSchedulerConfig):
    """Population Based Training scheduler settings."""

    type: str = schema_utils.StringOptions(options=["pbt"], default="pbt", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TIME_TOTAL_S,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    perturbation_interval: float = schema_utils.NonNegativeFloat(
        default=60.0,
        description=(
            "Models will be considered for perturbation at this interval of time_attr. Note that perturbation incurs "
            "checkpoint overhead, so you shouldn’t set this to be too frequent."
        ),
    )

    burn_in_period: float = schema_utils.NonNegativeFloat(
        default=60.0,
        description=(
            "Models will not be considered for perturbation before this interval of time_attr has passed. This "
            "guarantees that models are trained for at least a certain amount of time or timesteps before being "
            "perturbed."
        ),
    )

    hyperparam_mutations: Optional[Dict] = schema_utils.Dict(
        default=None,
        description=(
            "Hyperparams to mutate. The format is as follows: for each key, either a list, function, or a tune search "
            "space object (tune.loguniform, tune.uniform, etc.) can be provided. A list specifies an allowed set of "
            "categorical values. A function or tune search space object specifies the distribution of a continuous "
            "parameter. You must use tune.choice, tune.uniform, tune.loguniform, etc.. Arbitrary tune.sample_from "
            "objects are not supported. A key can also hold a dict for nested hyperparameters. You must specify at "
            "least one of hyperparam_mutations or custom_explore_fn. Tune will sample the search space provided by "
            "hyperparam_mutations for the initial hyperparameter values if the corresponding hyperparameters are not "
            "present in a trial’s initial config."
        ),
    )

    quantile_fraction: float = schema_utils.FloatRange(
        default=0.25,
        allow_none=False,
        min=0,
        max=0.5,
        description=(
            "Parameters are transferred from the top quantile_fraction fraction of trials to the bottom "
            "quantile_fraction fraction. Needs to be between 0 and 0.5. Setting it to 0 essentially implies doing no "
            "exploitation at all."
        ),
    )

    resample_probability: float = schema_utils.NonNegativeFloat(
        default=0.25,
        description=(
            "The probability of resampling from the original distribution when applying hyperparam_mutations. If not "
            "resampled, the value will be perturbed by a factor chosen from perturbation_factors if continuous, or "
            "changed to an adjacent value if discrete."
        ),
    )

    perturbation_factors: Tuple[float, float] = schema_utils.FloatRangeTupleDataclassField(
        default=(1.2, 0.8),
        allow_none=False,
        max=None,
        description=("Scaling factors to choose between when mutating a continuous hyperparameter."),
    )

    # TODO: Add schema support for Callable
    custom_explore_fn: Union[str, Callable] = schema_utils.String(
        description=(
            "You can also specify a custom exploration function. This function is invoked as f(config) after built-in "
            "perturbations from hyperparam_mutations are applied, and should return config updated as needed. You must "
            "specify at least one of hyperparam_mutations or custom_explore_fn."
        )
    )

    log_config: bool = schema_utils.Boolean(
        default=True,
        description=(
            "Whether to log the ray config of each model to local_dir at each exploit. Allows config schedule to be "
            "reconstructed."
        ),
    )

    require_attrs: bool = schema_utils.Boolean(
        default=True,
        description=(
            "Whether to require time_attr and metric to appear in result for every iteration. If True, error will be "
            "raised if these values are not present in trial result."
        ),
    )

    synch: bool = schema_utils.Boolean(
        default=False,
        description=(
            "If False, will use asynchronous implementation of PBT. Trial perturbations occur every "
            "perturbation_interval for each trial independently. If True, will use synchronous implementation of PBT. "
            "Perturbations will occur only after all trials are synced at the same time_attr every "
            "perturbation_interval. Defaults to False. See Appendix A.1 here https://arxiv.org/pdf/1711.09846.pdf."
        ),
    )


@register_scheduler_config("pbt_replay")
@dataclass
class PopulationBasedTrainingReplaySchedulerConfig(BaseSchedulerConfig):
    """Population Based Training Replay scheduler settings."""

    type: str = schema_utils.StringOptions(options=["pbt_replay"], default="pbt_replay", allow_none=False)

    # TODO: This should technically be a required paremeter. Do we need to add support for required params?
    policy_file: str = schema_utils.String(
        description=(
            "The PBT policy file. Usually this is stored in ~/ray_results/experiment_name/pbt_policy_xxx.txt where xxx "
            "is the trial ID."
        )
    )


@register_scheduler_config("pb2")
@dataclass
class PopulationBasedBanditsSchedulerConfig(BaseSchedulerConfig):
    """Population Based Bandits (PB2) scheduler settings."""

    type: str = schema_utils.StringOptions(options=["pb2"], default="pb2", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TIME_TOTAL_S,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    perturbation_interval: float = schema_utils.NonNegativeFloat(
        default=60.0,
        description=(
            "Models will be considered for perturbation at this interval of time_attr. Note that perturbation incurs "
            "checkpoint overhead, so you shouldn’t set this to be too frequent."
        ),
    )

    hyperparam_bounds: Optional[Dict] = schema_utils.Dict(
        default=None,
        description=(
            "Hyperparameters to mutate. The format is as follows: for each key, enter a list of the form [min, max] "
            "representing the minimum and maximum possible hyperparam values."
        ),
    )

    quantile_fraction: float = schema_utils.FloatRange(
        default=0.25,
        allow_none=False,
        min=0,
        max=0.5,
        description=(
            "Parameters are transferred from the top quantile_fraction fraction of trials to the bottom "
            "quantile_fraction fraction. Needs to be between 0 and 0.5. Setting it to 0 essentially implies doing no "
            "exploitation at all."
        ),
    )

    log_config: bool = schema_utils.Boolean(
        default=True,
        description=(
            "Whether to log the ray config of each model to local_dir at each exploit. Allows config schedule to be "
            "reconstructed."
        ),
    )

    require_attrs: bool = schema_utils.Boolean(
        default=True,
        description=(
            "Whether to require time_attr and metric to appear in result for every iteration. If True, error will be "
            "raised if these values are not present in trial result."
        ),
    )

    synch: bool = schema_utils.Boolean(
        default=False,
        description=(
            "If False, will use asynchronous implementation of PBT. Trial perturbations occur every "
            "perturbation_interval for each trial independently. If True, will use synchronous implementation of PBT. "
            "Perturbations will occur only after all trials are synced at the same time_attr every "
            "perturbation_interval. Defaults to False. See Appendix A.1 here https://arxiv.org/pdf/1711.09846.pdf."
        ),
    )


@register_scheduler_config("hb_bohb")
@dataclass
class BOHBSchedulerConfig(BaseSchedulerConfig):
    """Hyperband for BOHB scheduler settings."""

    type: str = schema_utils.StringOptions(options=["bohb"], default="bohb", allow_none=False)

    time_attr: str = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS),
        default=TRAINING_ITERATION,
        allow_none=False,
        description=(
            "A training result attr to use for comparing time. Note that you can pass in something non-temporal such as"
            " training_iteration as a measure of progress, the only requirement is that the attribute should increase "
            "monotonically."
        ),
    )

    metric: Optional[str] = schema_utils.StringOptions(
        options=list(DEFAULT_RESULT_KEYS) + [RAY_TUNE_DESULT_DEFAULT_METRIC],
        default=None,
        description=(
            "The training result objective value attribute. Stopping procedures will use this attribute. If None but a "
            "mode was passed, the ray.tune.result.DEFAULT_METRIC will be used per default."
        ),
    )

    mode: Optional[str] = schema_utils.StringOptions(
        options=["min", "max"],
        default=None,
        description=(
            "One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute."
        ),
    )

    max_t: int = schema_utils.PositiveInteger(
        default=81,
        description=(
            "max time units per trial. Trials will be stopped after max_t time units (determined by time_attr) have "
            "passed."
        ),
    )

    reduction_factor: int = schema_utils.NonNegativeFloat(
        default=3, description=("Used to set halving rate and amount. This is simply a unit-less scalar.")
    )

    stop_last_trials: bool = schema_utils.Boolean(
        default=True, description=("Whether to terminate the trials after reaching max_t. Defaults to True.")
    )


# TODO: Double-check support for this
@register_scheduler_config("fifo")
@dataclass
class FIFOSchedulerConfig(BaseSchedulerConfig):
    """FIFO trial scheduler settings."""

    type: str = schema_utils.StringOptions(options=["fifo"], default="fifo", allow_none=False)


# TODO: Double-check support for this as well as whether Callable args work properly
@register_scheduler_config("resource_changing")
@dataclass
class ResourceChangingSchedulerConfig(BaseSchedulerConfig):
    """Resource changing scheduler settings."""

    type: str = schema_utils.StringOptions(options=["resource_changing"], default="resource_changing", allow_none=False)

    base_scheduler: Union[str, None, Callable] = schema_utils.String(
        description=("The scheduler to provide decisions about trials. If None, a default FIFOScheduler will be used.")
    )

    resources_allocation_function: Union[str, Callable] = schema_utils.String(
        description=(
            "The callable used to change live trial resource requiements during tuning. This callable will be called on"
            " each trial as it finishes one step of training. The callable must take four arguments: TrialRunner, "
            "current Trial, current result dict and the ResourceChangingScheduler calling it. The callable must return "
            "a PlacementGroupFactory, Resources, dict or None (signifying no need for an update). If "
            "resources_allocation_function is None, no resource requirements will be changed at any time. By default, "
            "DistributeResources will be used, distributing available CPUs and GPUs over all running trials in a robust"
            " way, without any prioritization."
        )
    )


def get_scheduler_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for scheduler_config in scheduler_config_registry:
        scheduler_cls = scheduler_config_registry[scheduler_config]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(scheduler_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        preproc_cond = schema_utils.create_cond(
            {"type": scheduler_config},
            other_props,
        )
        conds.append(preproc_cond)
    return conds


def SchedulerDataclassField(default={"type": "async_hyperband"}, description="Hyperopt scheduler settings."):
    """Custom dataclass field that when used inside of a dataclass will allow any scheduler in
    `ludwig.schema.hyperopt.scheduler.scheduler_registry`.

    Sets default optimizer to 'async_hyperband'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load scheduler from registry with given params. (default: {"type": "async_hyperband"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """

    class OptimizerMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict to a valid optimizer from
        `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema
        for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if "type" in value and value["type"] in scheduler_config_registry:
                    opt = scheduler_config_registry[value["type"].lower()][1]
                    try:
                        return opt.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for optimizer: {value}, see `{opt}` definition. Error: {e}"
                        )
                raise ValidationError(
                    f"Invalid params for optimizer: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            # Note that this uses the same conditional pattern as combiners:
            return {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(scheduler_config_registry.keys()),
                        "default": default["type"],
                        "description": "The type of scheduler to use during hyperopt",
                    },
                },
                "title": "scheduler_options",
                "allOf": get_scheduler_conds(),
                "required": ["type"],
                "description": description,
            }

    if not isinstance(default, dict) or "type" not in default or default["type"] not in scheduler_config_registry:
        raise ValidationError(f"Invalid default: `{default}`")
    try:
        opt = scheduler_config_registry[default["type"].lower()]
        load_default = opt.Schema().load(default)
        dump_default = opt.Schema().dump(default)

        return field(
            metadata={
                "marshmallow_field": OptimizerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                    metadata={"description": description},
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(
            f"Unsupported scheduler type: {default['type']}. See scheduler_config_registry. Details: {e}"
        )