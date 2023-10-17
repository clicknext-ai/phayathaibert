from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.checkpointing import load_custom_state
from accelerate.state import PartialState
from accelerate.utils.other import is_compiled_module
from accelerate.utils import (
    is_torch_version,
    convert_outputs_to_fp32,
    has_transformer_engine_layers,
    convert_model,
    is_deepspeed_available,
    is_fp8_available,
    is_tpu_available,
    is_xpu_available,
    DynamoBackend,
    MODEL_NAME,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCALER_NAME,
    RNG_STATE_NAME
)
from types import MethodType
from collections.abc import Container
import torch, inspect, os, numpy, random

if is_deepspeed_available():
    from deepspeed import DeepSpeedEngine
    from accelerate.utils import DeepSpeedSchedulerWrapper

if is_fp8_available():
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast

if is_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

logger = get_logger(__name__)

class CustomAccelerator(Accelerator):

    def prepare_model(
        self,
        model: torch.nn.Module,
        device_placement: bool = None,
        evaluation_mode: bool = False
    ):
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
        old_model_type = type(model)
        try:
            model_index = self._models.index(model)
            reprepare = True
        except ValueError:
            self._models.append(model)
            model_index = -1
            reprepare = False

        # We check only for models loaded with `accelerate`
        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = False
        for m in model.modules():
            if hasattr(m, "hf_device_map"):
                has_hf_device_map = True
                break

        if (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)) and getattr(
            model, "hf_device_map", False
        ):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1 and self.distributed_type != DistributedType.NO:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices in any distributed mode."
                    " In order to use 8-bit models that have been loaded across multiple GPUs the solution is to use Naive Pipeline Parallelism."
                    " Therefore you should not specify that you are under any distributed regime in your accelerate config."
                )
            current_device = list(model_devices)[0]
            current_device_index = current_device.index if isinstance(current_device, torch.device) else current_device

            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not has_hf_device_map:
            model = model.to(self.device)

        if not evaluation_mode:
            if self.distributed_type in (DistributedType.MULTI_GPU, DistributedType.MULTI_XPU):
                if any(p.requires_grad for p in model.parameters()):
                    kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                    model = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
                    )
            elif self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

                # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
                # don't wrap it again
                if type(model) != FSDP:
                    self.state.fsdp_plugin.set_auto_wrap_policy(model)
                    fsdp_plugin = self.state.fsdp_plugin
                    kwargs = {
                        "sharding_strategy": fsdp_plugin.sharding_strategy,
                        "cpu_offload": fsdp_plugin.cpu_offload,
                        "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                        "backward_prefetch": fsdp_plugin.backward_prefetch,
                        "mixed_precision": fsdp_plugin.mixed_precision_policy,
                        "ignored_modules": fsdp_plugin.ignored_modules,
                        "device_id": self.device,
                    }
                    signature = inspect.signature(FSDP.__init__).parameters.keys()
                    if "limit_all_gathers" in signature:
                        kwargs["limit_all_gathers"] = fsdp_plugin.limit_all_gathers
                    if "use_orig_params" in signature:
                        kwargs["use_orig_params"] = fsdp_plugin.use_orig_params
                    model = FSDP(model, **kwargs)
                self._models[model_index] = model
            elif self.distributed_type == DistributedType.MULTI_CPU:
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        if self.native_amp:
            model._original_forward = model.forward
            if self.mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
                model.forward = MethodType(torch.cuda.amp.autocast(dtype=torch.float16)(model.forward.__func__), model)
            elif self.mixed_precision == "bf16" and self.distributed_type != DistributedType.TPU:
                model.forward = MethodType(
                    torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(model.forward.__func__), model
                )
            else:
                model.forward = MethodType(torch.cuda.amp.autocast()(model.forward.__func__), model)
            model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
        elif self.mixed_precision == "fp8":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            cuda_device_capacity = torch.cuda.get_device_capability()
            fp8_enabled = cuda_device_capacity[0] >= 9 or (
                cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
            )
            if not fp8_enabled:
                logger.warn(
                    f"The current device has compute capability of {cuda_device_capacity} which is "
                    "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                    "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                )
            model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if not evaluation_mode:
            if self.distributed_type == DistributedType.TPU and self.state.fork_launched:
                model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        new_model_type = type(model)
        if self.is_main_process and new_model_type is not old_model_type:
            model_type_name = new_model_type.__name__
            if reprepare:
                print(f"Reprepared '{model_type_name}'")
            else:
                print(f"Wrapped model into '{model_type_name}'")
        return model

    def load_state(
        self,
        input_dir: str,
        *,
        ignore_missing: Container[str] = (),
        **load_model_func_kwargs
    ):
        # Check if folder exists
        input_dir = os.path.expanduser(input_dir)
        if not os.path.isdir(input_dir):
            raise ValueError(f"Tried to find {input_dir} but folder does not exist")
        logger.info(f"Loading states from {input_dir}")

        # Load the models taking care of FSDP and DeepSpeed nuances
        models = []
        for i, model in enumerate(self._models):
            if self.distributed_type == DistributedType.FSDP:
                logger.info("Loading FSDP model")
                self.state.fsdp_plugin.load_model(self, model, input_dir, i)
                logger.info(f"FSDP Model loaded from input dir {input_dir}")
            elif self.distributed_type == DistributedType.DEEPSPEED:
                logger.info("Loading DeepSpeed Model and Optimizer")
                ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
                model.load_checkpoint(input_dir, ckpt_id, **load_model_func_kwargs)
                logger.info(f"DeepSpeed Model and Optimizer loaded from input dir {os.path.join(input_dir, ckpt_id)}")
            elif self.distributed_type == DistributedType.MEGATRON_LM:
                logger.info("Loading Megatron-LM Model, Optimizer and Scheduler")
                model.load_checkpoint(input_dir)
                logger.info(f"Megatron-LM Model , Optimizer and Scheduler loaded from input dir {input_dir}")
            else:
                models.append(model)

        # Load the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        if self.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(self._optimizers):
                logger.info("Loading FSDP Optimizer")
                self.state.fsdp_plugin.load_optimizer(self, opt, self._models[i], input_dir, i)
                logger.info(f"FSDP Optimizer loaded from input dir {input_dir}")
        elif self.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = self._optimizers

        # Load the lr schedulers taking care of DeepSpeed nuances
        schedulers = []
        if self.distributed_type == DistributedType.DEEPSPEED:
            for i, scheduler in enumerate(self._schedulers):
                if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                    continue
                schedulers.append(scheduler)
        elif self.distributed_type not in [DistributedType.MEGATRON_LM]:
            schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._load_model_state_pre_hook.values():
            hook(models, input_dir)

        map_location = load_model_func_kwargs.pop("map_location", None)
        if map_location is None:
            if self.num_processes > 1 and self.distributed_type == DistributedType.MULTI_GPU:
                map_location = "on_device"
            else:
                map_location = "cpu"

        load_accelerator_state(
            input_dir,
            models,
            optimizers,
            schedulers,
            self.state.process_index,
            self.scaler,
            map_location,
            ignore_missing=ignore_missing,
            **load_model_func_kwargs,
        )
        custom_checkpoints = [f for f in os.listdir(input_dir) if "custom_checkpoint" in f]
        if len(custom_checkpoints) != len(self._custom_objects):
            err = "Warning! Number of found checkpoints does not match the number of registered objects:"
            err += f"\n\tFound checkpoints: {len(custom_checkpoints)}"
            err += f"\n\tRegistered objects: {len(self._custom_objects)}\nSkipping."
            logger.warning(err)
        else:
            logger.info(f"Loading in {len(custom_checkpoints)} custom states")
            for index, obj in enumerate(self._custom_objects):
                load_custom_state(obj, input_dir, index)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

        is_compiled = is_compiled_module(model)
        if is_compiled:
            compiled_model = model
            model = model._orig_mod

        if is_deepspeed_available():
            options += (DeepSpeedEngine,)

        while isinstance(model, options):
            model = model.module

        if not keep_fp32_wrapper:
            forward = getattr(model, "forward")
            original_forward = model.__dict__.pop("_original_forward", None)
            if original_forward is not None:
                while hasattr(forward, "__wrapped__"):
                    forward = forward.__wrapped__
                    if forward == original_forward:
                        break
                model.forward = MethodType(forward, model)
            if getattr(model, "_converted_to_transformer_engine", False):
                convert_model(model, to_transformer_engine=False)

        if is_compiled:
            compiled_model._orig_mod = model
            model = compiled_model

        return model

def load_accelerator_state(
    input_dir,
    models,
    optimizers,
    schedulers,
    process_index,
    scaler=None,
    map_location=None,
    *,
    ignore_missing,
    **load_model_func_kwargs,
):
    if map_location not in [None, "cpu", "on_device"]:
        raise TypeError(
            "Unsupported optimizer map location passed, please choose one of `None`, `'cpu'`, or `'on_device'`"
        )
    if map_location is None:
        map_location = "cpu"
    elif map_location == "on_device":
        map_location = PartialState().device

    # Model states
    try:
        for i, model in enumerate(models):
            weights_name = f"{MODEL_NAME}.bin" if i == 0 else f"{MODEL_NAME}_{i}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            model.load_state_dict(torch.load(input_model_file, map_location=map_location), **load_model_func_kwargs)
        logger.info("All model weights loaded successfully")
    except Exception:
        logger.info("Could not load model weights")
        if "model" not in ignore_missing:
            raise

    # Optimizer states
    try:
        for i, optimizer in enumerate(optimizers):
            optimizer_name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
            input_optimizer_file = os.path.join(input_dir, optimizer_name)
            optimizer_state = torch.load(input_optimizer_file, map_location=map_location)
            optimizer.load_state_dict(optimizer_state)
        logger.info("All optimizer states loaded successfully")
    except Exception:
        logger.info("Could not load optimizer states")
        if "optimizer" not in ignore_missing:
            raise

    # Scheduler states
    try:
        for i, scheduler in enumerate(schedulers):
            scheduler_name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
            input_scheduler_file = os.path.join(input_dir, scheduler_name)
            scheduler.load_state_dict(torch.load(input_scheduler_file))
        logger.info("All scheduler states loaded successfully")
    except Exception:
        logger.info("Could not load scheduler states")
        if "scheduler" not in ignore_missing:
            raise

    # GradScaler state
    try:
        if scaler is not None:
            input_scaler_file = os.path.join(input_dir, SCALER_NAME)
            scaler.load_state_dict(torch.load(input_scaler_file))
            logger.info("GradScaler state loaded successfully")
    except Exception:
        logger.info("Could not load GradScaler state")
        if "scaler" not in ignore_missing:
            raise

    # Random states
    try:
        states = torch.load(os.path.join(input_dir, f"{RNG_STATE_NAME}_{process_index}.pkl"))
        random.setstate(states["random_state"])
        numpy.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(states["torch_xpu_manual_seed"])
        else:
            torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
        if is_tpu_available():
            xm.set_rng_state(states["xm_seed"])
        logger.info("All random states loaded successfully")
    except Exception:
        logger.info("Could not load random states")
        if "rng_state" not in ignore_missing:
            raise
