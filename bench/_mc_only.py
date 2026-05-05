import os, sys
sys.path.insert(0, '/home/ubuntu/efs/zm/kernels/release/python')
import load_module
import torch, torch.distributed as dist
os.environ["OSGC_BIND_RETAINED_HANDLE"] = "1"
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
mod = load_module.load("dispatch_gemm")
mc = mod.DistBuffer((1, 1, 2), dtype=torch.int32, local_rank=local_rank, local_world_size=world_size, multicast=True)
print(f"lr{local_rank} mc OK", flush=True)
