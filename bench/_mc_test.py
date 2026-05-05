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
print(f"[order-test] lr{local_rank} OSGC_BIND={os.environ.get('OSGC_BIND_RETAINED_HANDLE')}", flush=True)

# Mirror the bench's allocation order: 4 non-mc tensors, then mc.
H = 7168
ts = []
ts.append(mod.DistBuffer((512, H), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=world_size, multicast=False))
print(f"[order-test] lr{local_rank} t0 ok", flush=True)
ts.append(mod.DistBuffer((512, H), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=world_size, multicast=False))
print(f"[order-test] lr{local_rank} t1 ok", flush=True)
ts.append(mod.DistBuffer((world_size, 4, 1), dtype=torch.int32, local_rank=local_rank, local_world_size=world_size, multicast=False))
print(f"[order-test] lr{local_rank} t2 ok", flush=True)

# Now the multicast one
mc = mod.DistBuffer((1, 1, 2), dtype=torch.int32, local_rank=local_rank, local_world_size=world_size, multicast=True)
print(f"[order-test] lr{local_rank} mc ok", flush=True)
