# file: run_adv_parallel.py
import torch
from adversarial.embedding import adv_emb_attack
from multiprocessing import Process

def run_attack(strength, out, device):
    adv_emb_attack(
        "/scratch/qilong3/transferattack/results/hiddencnn30db_Attacks_hiddencnn30db_1modelssdxlvae/images/watermarked_before_attack",
        encoder="sdxlvae",
        strength=strength,
        output_path=out,
        device=torch.device(device),
    )

if __name__ == "__main__":
    cfgs = [
        dict(strength=31, out="data/out", device="cuda:1"),
        dict(strength=12, out="data/out", device="cuda:2"),
    ]

    procs = []
    for c in cfgs:
        p = Process(target=run_attack, args=(c["strength"], c["out"], c["device"]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
