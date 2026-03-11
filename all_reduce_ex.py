import os
import torch
import torch.distributed as dist
import argparse

def run_handshake(args):
    # 1. THE RENDEZVOUS
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{args.master_ip}:{args.master_port}',
        rank=args.rank,
        world_size=args.world_size
    )

    identities = {0: "Namer", 1: "Se'er", 2: "Shape-Shifter", 3: "Healer"}
    my_identity = identities[args.rank]
    print(f"[{my_identity}] Awoke")
    # --- PHASE 1: SCATTER (Distributing unique keys) ---
    received_key = torch.zeros(1) 
    if args.rank == 0:
        # scatter_list = [torch.tensor([10.0]), torch.tensor([20.0]), 
        #                 torch.tensor([30.0]), torch.tensor([40.0])]
        scatter_list = [torch.tensor([10.0]), torch.tensor([20.0])]
    else:
        scatter_list = None

    dist.scatter(received_key, scatter_list, src=0)
    print(f"[{my_identity}] Received key: {received_key.item()}")

    #All Reduce -- Add Up The Secret Keys
    dist.all_reduce(received_key, op=dist.ReduceOp.SUM)

    new_key = received_key/args.world_size

    print(f"[{my_identity}] All Reduced: {new_key}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--master_ip", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="23456")
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    
    run_handshake(args)