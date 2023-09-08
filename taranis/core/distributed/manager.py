import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record as torch_record
from torch.nn.parallel import DistributedDataParallel

log = logging.getLogger("distributed")


class DistributedProcessGroup:
    """Helper to manage distributed training setup.
    All its operation as set to be noop if a single-gpu is used.

    """

    INSTANCE = None

    def __init__(self, backend="nccl"):
        # SLURM
        slurm_rank = int(os.environ.get("SLURM_LOCALID", -1))
        slurm_grank = int(os.environ.get("SLURM_PROCID", -1))
        slurm_world_size = int(os.environ.get("SLURM_NTASKS", -1))

        # Use Pytorch environment first
        self.__rank = int(os.environ.get("LOCAL_RANK", slurm_rank))
        self.__group = int(os.environ.get("GROUP_RANK", -1))
        self.__grank = int(os.environ.get("RANK", slurm_grank))
        self.__world = int(os.environ.get("WORLD_SIZE", slurm_world_size))

        # In case SLURM was used as the source make sure pytorch
        # expected variables are available
        os.environ["LOCAL_RANK"] = str(self.__rank)
        os.environ["RANK"] = str(self.__grank)
        os.environ["WORLD_SIZE"] = str(self.__world)

        self._log_prefix = f"[{self.__rank}][{self.__group}]"
        if self.__rank < 0:
            self._log_prefix = ""

        if self.__rank >= 0:
            log.info("%s Initializing process group", self._log_prefix)
            dist.init_process_group(
                backend,
                init_method="env://",
                # This should not be necessary
                world_size=self.__world,
                rank=self.__rank,
            )
            log.info("%s Process group initialized", self._log_prefix)

        assert DistributedProcessGroup.INSTANCE is None
        DistributedProcessGroup.INSTANCE = self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.shutdown()

    def __del__(self):
        self.shutdown()

    def barrier(self):
        if self.rank >= 0:
            dist.barrier()

    @property
    def info(self):
        """return a string identifying the current worker"""
        return self._log_prefix

    @property
    def group(self):
        """Current node this worker belongs"""
        return self.__group

    @property
    def global_rank(self):
        return self.__grank

    @property
    def rank(self):
        """Return the current rank of our script -1 if running as a single GPU"""
        return self.__rank

    @property
    def has_weight_autority(self):
        """Is the worker controlling/checkpointing weights"""
        return self.__grank <= 0

    @property
    def has_dataset_autority(self):
        """Is the worker downloading/setting up the dataset locally"""
        return self.__rank <= 0

    @property
    def has_metric_autority(self):
        """Is the worker computing metrics"""
        return self.has_weight_autority

    def shutdown(self):
        """Close the process group if running in distributed mode"""
        if self.__rank >= 0:
            log.info("Process group shutdown")
            dist.destroy_process_group()

        DistributedProcessGroup.INSTANCE = None

    @property
    def device_id(self):
        """Return the device id this script should use"""
        if self.rank < 0:
            return -1

        return self.rank % torch.cuda.device_count()

    @property
    def device(self):
        """Return the device this script should use"""
        if self.rank < 0:
            return torch.device("cuda")

        return torch.device(f"cuda:{self.device_id}")


def __get_attr(name, default):
    group = DistributedProcessGroup.INSTANCE
    if group is None:
        return default

    return getattr(group, name)


def barrier():
    """block until all workers reach this"""
    return __get_attr("barrier", lambda: True)()


def has_weight_autority():
    """Returns true if the current worker has control over saving the weights"""
    return __get_attr("has_weight_autority", True)


def has_dataset_autority():
    """Returns true if the current worker has control over downloading the dataset"""
    return __get_attr("has_dataset_autority", True)


def has_metric_autority():
    """Returns true if the current worker has control over saving the metrics"""
    return __get_attr("has_metric_autority", True)


def rank():
    """Returns local rank"""
    return __get_attr("rank", -1)


def enabled():
    return DistributedProcessGroup.INSTANCE is not None


def grank():
    """Returns global rank"""
    return __get_attr("global_rank", -1)


def world_size():
    """Returns global rank"""
    return __get_attr("world_size", 1)


def device_id():
    """Returns current device_id"""
    return __get_attr("device_id", 0)


def fetch_device():
    """Set the default device to CPU if cuda is not available"""
    default = "cpu"
    if torch.cuda.is_available():
        default = "cuda"

    if rank() >= 0:
        return torch.device(f"{default}:{device_id()}")

    return torch.device(default)


def device():
    """Returns current device"""
    return __get_attr("device", fetch_device())


def dataparallel(model, device=None):
    """Wrap the model to make it parallel if rank is not none"""
    if rank() >= 0:
        log.info("enabling multi-gpu %s", device_id())
        return DistributedDataParallel(
            model, device_ids=[device_id()], output_device=device_id()
        )

    return model


def record(fn, error_handler=None):
    """Decorator that helps record an exception in a distributed setup"""
    if rank() >= 0:
        return torch_record(fn, error_handler)

    return fn


def all_reduce(value, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if rank() >= 0:
        return dist.all_reduce(value, op=op, group=group, async_op=async_op)


def reduce(value, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
    if rank() >= 0:
        return dist.reduce(value, dst, op=op, group=group, async_op=async_op)


@contextmanager
def distributed(*args, enabled=True, **kwargs):
    obj = None
    if enabled:
        obj = DistributedProcessGroup(*args, **kwargs)

    yield obj
