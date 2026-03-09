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

    # --- PHASE 1: SCATTER (Distributing unique keys) ---
    received_key = torch.zeros(1) 
    if args.rank == 0:
        scatter_list = [torch.tensor([10.0]), torch.tensor([20.0]), 
                        torch.tensor([30.0]), torch.tensor([40.0])]
    else:
        scatter_list = None

    dist.scatter(received_key, scatter_list, src=0)
    print(f"[{my_identity}] Received key: {received_key.item()}")

    # --- PHASE 2: GATHER (Reporting back to the Namer) ---
    # Everyone prepares to send their key back (perhaps modified)
    report_tensor = received_key.clone()
    
    if args.rank == 0:
        # Namer prepares a list to catch all 4 reports
        gather_list = [torch.zeros(1) for _ in range(args.world_size)]
    else:
        gather_list = None

    dist.gather(report_tensor, gather_list, dst=0)

    if args.rank == 0:
        print(f"[{my_identity}] I have gathered the reports: {[t.item() for t in gather_list]}")

    # --- PHASE 3: BROADCAST (One message to rule them all) ---
    # The Namer creates a "Confirmation Code"
    conf_code = torch.tensor([99.0]) if args.rank == 0 else torch.zeros(1)

    # Everyone calls broadcast; Rank 0 provides the data, others receive
    dist.broadcast(conf_code, src=0)
    print(f"[{my_identity}] Received broadcast confirmation: {conf_code.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    parser.add_argument("--master_ip", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="23456")
    parser.add_argument("--world_size", type=int, default=4)
    args = parser.parse_args()
    
    run_handshake(args)