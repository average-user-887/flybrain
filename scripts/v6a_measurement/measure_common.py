"""Shared helpers for the v6a measurement scripts."""
import os
import subprocess


def gpu_identity():
    """Which physical card this process runs on, for every receipt."""
    import cupy
    dev = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    name = props['name']
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,name,pci.bus_id,memory.used,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_bus_id',
                           '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    return dict(cuda_device_name=name.decode() if isinstance(name, bytes) else name,
                cuda_pci_bus_id=cupy.cuda.runtime.deviceGetPCIBusId(dev.id),
                CUDA_DEVICE_ORDER=os.environ.get('CUDA_DEVICE_ORDER'),
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                pid=os.getpid(), nvidia_smi_gpus=smi, nvidia_smi_compute_apps=apps)
