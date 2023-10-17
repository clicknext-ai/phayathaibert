from .config import Config, Layer
from typing import TypedDict, TypeVar, Required
from collections.abc import Iterable, Hashable, Set
from torch.nn import Module, Parameter
from transformers import PreTrainedModel
from functools import reduce

def get_module(model: PreTrainedModel, name: str) -> Module | Parameter:
    return reduce(getattr, name.split('.'), model)

def get_params(model: PreTrainedModel, name: str):
    module = get_module(model, name)
    if isinstance(module, Parameter):
        yield module
    else:
        for param in module.parameters():
            yield param

T = TypeVar('T', bound=Hashable)
class OrderedSet(Set[T]):
    def __init__(self, iterable: Iterable[T]):
        self.__dict = dict.fromkeys(iterable)
    def __contains__(self, item):
        return item in self.__dict
    def __iter__(self):
        return iter(self.__dict)
    def __len__(self):
        return len(self.__dict)
    def __repr__(self):
        return f"{type(self).__name__}({', '.join(repr(item) for item in self)})"

def get_layer_params(model: PreTrainedModel, layer: Layer):
    include = OrderedSet(param for name in layer.include for param in get_params(model, name))
    exclude = {param for name in layer.exclude for param in get_params(model, name)} if layer.exclude is not None else set()
    return include - exclude

def check_layer_is_exhaustive(model: PreTrainedModel, config: Config):
    accounted_for = {param for layer in config.layer_config.layers for param in get_layer_params(model, layer)}
    num_unaccounted_for = len(set(model.parameters()) - accounted_for)
    if num_unaccounted_for != 0:
        raise ValueError(
            f"'layer' defined in {config.script_config.config_file} is not exhaustive."
            f" {num_unaccounted_for} parameters are not accounted for."
        )

class ParameterGroup(TypedDict):
    params: Required[list[Parameter]]
    lr: Required[float]

def get_optimizer_param_groups(model: PreTrainedModel, config: Config):
    optimizer_config = config.optimizer_config
    if optimizer_config.layer_lr_decay_factor is None:
        return model.parameters()
    else:
        # Discriminative fine-tuning
        params: list[ParameterGroup] = []
        param_set: set[Parameter] = set()
        current_lr = optimizer_config.peak_lr
        decay_factor = optimizer_config.layer_lr_decay_factor
        for layer in config.layer_config.layers:
            param_group = ParameterGroup(params=[], lr=current_lr)
            for param in get_layer_params(model, layer):
                if param not in param_set:
                    param_group["params"].append(param)
                    param_set.add(param)
            params.append(param_group)
            current_lr /= decay_factor
        return params
