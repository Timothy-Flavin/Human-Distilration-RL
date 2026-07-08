import numpy as np
import matplotlib.pyplot as plt
import os

modes = ["RL", "BC", "RL_BC", "RL_Naive_BC"]
scales = [1.0, 100.0]

if not os.path.exists("test_results"):
    print("test_results directory not found.")
    exit(1)

for scale in scales:
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, arch in enumerate(["cql", "rcql"]):
        for mode in modes:
            try:
                kl_data = np.load(f"test_results/{arch}_{mode}_scale_{scale}_kl.npy")
                td_data = np.load(f"test_results/{arch}_{mode}_scale_{scale}_td.npy")
                bc_data = np.load(f"test_results/{arch}_{mode}_scale_{scale}_bc.npy")
                
                kl_mean = np.mean(kl_data, axis=0)
                kl_std = np.std(kl_data, axis=0)
                axs[i, 0].plot(kl_mean, label=mode)
                axs[i, 0].fill_between(range(len(kl_mean)), kl_mean-kl_std, kl_mean+kl_std, alpha=0.2)
                
                td_mean = np.mean(td_data, axis=0)
                axs[i, 1].plot(td_mean, label=mode)
                
                bc_mean = np.mean(bc_data, axis=0)
                axs[i, 2].plot(bc_mean, label=mode)
            except FileNotFoundError:
                print(f"Data not found for {arch} {mode} scale {scale}")
                continue
                
        axs[i, 0].set_title(f"{arch.upper()} KL Divergence (Scale: {scale})")
        axs[i, 0].set_ylabel("KL Div")
        axs[i, 0].set_xlabel("Iterations")
        axs[i, 0].set_ylim(0, 5)  # Capping the y-axis to 5
        axs[i, 0].legend()
        
        axs[i, 1].set_title(f"{arch.upper()} TD Loss (Scale: {scale})")
        axs[i, 1].set_xlabel("Iterations")
        
        axs[i, 2].set_title(f"{arch.upper()} BC Loss (Scale: {scale})")
        axs[i, 2].set_xlabel("Iterations")
        
    plt.tight_layout()
    filepath = f"test_results/integration_results_scale_{scale}_ylim5.png"
    plt.savefig(filepath)
    print(f"Saved re-plotted figure to {filepath}")
