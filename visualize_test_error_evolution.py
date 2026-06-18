import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_data_series(root_dir, noise_std, repeats=10, alg='sgd', lr=0.01, iter=20000):
    # Determine the file name
    filename = f"record_{alg}_{iter}iter_{lr}lr_nstd{noise_std}.csv"
    if alg == 'localsgd':
        filename = f"record_{alg}_M8_R200_K100_lr{lr}_nstd{noise_std}.csv"
    
    all_dfs = []
    
    for r in range(1, repeats + 1):
        file_path = os.path.join(root_dir, f"nstd_{noise_std}", f"r{r}", "records", filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # Ensure sorting by iteration
            if alg == 'localsgd' and 'step' in df.columns:
                df = df.rename(columns={'step': 'iteration'})
            df = df.sort_values('iteration')
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return all_dfs

def process_stability_stats(all_dfs):
    if not all_dfs:
        return None
        
    combined_df = pd.concat(all_dfs)

    # Filter out unwanted iterations
    exclude_iterations = [200, 300, 400, 600, 700, 800, 900]
    combined_df = combined_df[~combined_df['iteration'].isin(exclude_iterations)]
    
    # Group by iteration and compute mean and std for stability
    grouped = combined_df.groupby('iteration')['test_acc'].agg(['mean', 'std'])
    
    return grouped

def plot_stability_evolution(data_store, save_path="stability_noise_evolution.png"):
    fig, ax = plt.subplots(figsize=(8,6))
    
    colors = {0.0: 'tab:blue', 0.5: 'tab:purple', 1.0: 'tab:green', 1.5: 'tab:orange'}
    markers = {0.0: 'o', 0.5: 's', 1.0: '^', 1.5: 'D'}
    
    for w, stats in data_store.items():
        if stats is None or stats.empty:
            continue
            
        iterations = stats.index.values
        
        # Determine sampling for error bars to avoid clutter
        # If we have many points, we show error bars every N points
        n_points = len(iterations)
        err_every = max(1, n_points // 20)
        
        ax.errorbar(iterations, stats['mean'], yerr=stats['std'], 
                label=f'std={w}', 
                    color=colors.get(w, 'black'),
                    marker=markers.get(w, '.'),
                    markevery=err_every,
                    errorevery=err_every,
                    capsize=8,
                    linestyle='-',
                    linewidth=4,
                    elinewidth=1)
    
    ax.set_xlabel('Iterations', fontsize=20)
    ax.set_ylabel('Test Accuracy', fontsize=20)
    # ax.set_title('Stability Evolution for Different Noise Std Values')
    
    ax.legend(fontsize=20, loc='lower right')
    ax.grid(True)
    
    # Format Y-axis to 2 decimal places
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    # Format X-axis to scientific notation
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=600)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Configuration
    root_dir = r"E:\Papers\SGD with bias\Experiment\Exp\localSGD\mnist"
    noise_std_list = [0.0, 0.5, 1.0, 1.5]
    repeats = 5
    alg = 'localsgd'
    lr = 0.001
    
    data_store = {}
    
    for noise_std in noise_std_list:
        print(f"Loading data for noise_std={noise_std}...")
        all_dfs = load_data_series(root_dir, noise_std, repeats, alg=alg, lr=lr)
        data_store[noise_std] = process_stability_stats(all_dfs)
    
    print("Plotting...")
    plot_stability_evolution(data_store, save_path=f"0617\{alg}_test_acc_noise_mnist.png")
