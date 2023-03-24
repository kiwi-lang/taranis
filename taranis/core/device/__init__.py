import torch
import os

CUDA = torch.cuda.is_available()


def sync():
    if CUDA:
        torch.cuda.synchronize()


def cpu_count():
    cpu = os.environ.get("SLURM_CPUS_PER_TASK")

    if cpu is not None:
        return int(cpu)

    # Windows does not have that function
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    return torch.multiprocessing.cpu_count()
