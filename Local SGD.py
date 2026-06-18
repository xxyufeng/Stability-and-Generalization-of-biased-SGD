import os
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if k in os.environ:
        del os.environ[k]

os.environ["RAY_raylet_ip_address"] = "127.0.0.1"
os.environ["RAY_node_ip_address"] = "127.0.0.1"

os.environ["RAY_raylet_start_wait_time_s"] = "120" 
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import pandas as pd
import time
import copy

from dataset_and_model import load_data, load_model

############################
# Utils & Logger 
############################

# Creates "neighboring datasets" for stability analysis.
# A neighbor dataset S' differs from S by exactly one example.
# This function generates 'n_pairs' of such neighbor datasets.
def create_neibordataset(dataset, total_samples, n_pairs, seed=None):
    if seed is not None:
        np.random.seed(42+seed)
    indices = list(range(len(dataset)))
    sample_count = min(len(dataset), total_samples+n_pairs)
    indices = np.random.choice(indices, sample_count, replace=False)
    
    print("----Creating neighboring dataset----")
    print(f"Total used samples: {len(indices)}")

    removed_indices = np.random.choice(indices, n_pairs, replace=False)
    neighbordataset_1 = [i for i in indices if i not in removed_indices]
    datasets_dict = {0: neighbordataset_1.copy()}
    
    replace_indices = np.random.choice(neighbordataset_1, n_pairs, replace=False)
    for i in range(n_pairs):
        new_subset = neighbordataset_1.copy()
        try:
            idx = new_subset.index(replace_indices[i])
            new_subset[idx] = removed_indices[i]
            datasets_dict[i+1] = new_subset
        except ValueError:
            pass # Should not happen given logic
    return datasets_dict

class Logger:
    def __init__(self, log_dir, log_filename):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_filename)

    def update(self, step, t_loss, t_acc, te_loss, te_acc, gap, stab):
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"Step: {step}, "
                    f"TrLoss: {t_loss:.4f}, TrAcc: {t_acc:.4f}, "
                    f"TeLoss: {te_loss:.4f}, TeAcc: {te_acc:.4f}, "
                    f"Gap: {gap:.4f}, Stab: {stab:.6f}\n")

class DataRecorder:
    def __init__(self, rec_dir, filename):
        self.rec_dir = rec_dir
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir, exist_ok=True)
        self.data = []
        self.save_path = os.path.join(rec_dir, filename)

    def update(self, rec_dict):
        self.data.append(rec_dict)

    def save(self):
        if not self.data: return
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

class ModelCheckpoint:
    def __init__(self, checkpoint_dir, context_str):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.context = context_str

    def save(self, model, iteration):
        path = os.path.join(self.checkpoint_dir, f"{self.context}_iter{iteration}.pt")
        torch.save(model.state_dict(), path)

def run_inference(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4 if device == 'cuda' else 0)
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


############################
# Worker
############################

@ray.remote
class LocalSGDWorker:
    def __init__(self, worker_id, args, indices_dict, current_noise_std=0.0, current_seed=42):
        if args.device == 'cpu':
            torch.set_num_threads(1)
            
        self.worker_id = worker_id
        self.args = args
        self.current_noise_std = current_noise_std
        self.current_seed = current_seed
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        
        self.num_models = args.n_pairs + 1
        self.models = {i: load_model(args.model, args.loss, q=args.q).to(self.device) for i in range(self.num_models)}
        self.optimizers = {i: optim.SGD(self.models[i].parameters(), lr=args.lr) for i in range(self.num_models)}
        
        self.full_train = load_data(args.dataset, args.dataset_path, 'train')

        def _add_gaussian_noise(self, tensor, std=1, random_seed=42):
            noise = torch.randn(tensor.shape, generator=torch.Generator(device=self.device).manual_seed(random_seed)) * std
            tensor += noise
            tensor.clamp_(0, 255)
            return tensor

        temp_loader = DataLoader(self.full_train, batch_size=1000, num_workers=4 if self.device == 'cuda' else 0, shuffle=False)
        all_x, all_y = [], []
        for bx, by in temp_loader:
            if self.current_noise_std > 0:
                bx = _add_gaussian_noise(self, bx, std=self.current_noise_std, random_seed=self.current_seed+1)
            all_x.append(bx)
            all_y.append(by)
        self.cached_x = torch.cat(all_x)
        self.cached_y = torch.cat(all_y)
        
        self.indices_dict = {k: list(v) for k, v in indices_dict.items()}
        self.dataset_len = len(self.indices_dict[0])
        self.perm_indices = np.arange(self.dataset_len)
        self.curr_pos = 0
            
        torch.manual_seed(args.seed_base + worker_id)
        np.random.seed(args.seed_base + worker_id)
        np.random.shuffle(self.perm_indices)

    def _get_next_batch_indices(self):
        """Get next batch indices (Position Indices)"""
        if self.curr_pos >= self.dataset_len:
            np.random.shuffle(self.perm_indices)
            self.curr_pos = 0
            
        end_pos = min(self.curr_pos + self.args.batch_size, self.dataset_len)
        batch_positions = self.perm_indices[self.curr_pos : end_pos]
        self.curr_pos = end_pos
        return batch_positions

    def _fetch_data(self, model_idx, batch_positions):
        """Get real data for a specific model based on unified position indices"""
        # 1. Map to the real dataset indices for the specific model
        real_indices = [self.indices_dict[model_idx][p] for p in batch_positions]
        
        # 2. load data from cached tensors
        if len(real_indices) > 0:
            inputs = self.cached_x[real_indices]
            labels = self.cached_y[real_indices]
        else:
            inputs = torch.empty(0)
            labels = torch.empty(0)
            
        return inputs, labels

    def train_k_steps(self, K):
        for _ in range(K):
            batch_positions = self._get_next_batch_indices()
            for i in range(self.num_models):
                self.models[i].train()
                self.optimizers[i].zero_grad()
                
                inputs, labels = self._fetch_data(i, batch_positions)
                if inputs.size(0) == 0: continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs, loss = self.models[i](inputs, labels)
                loss.backward()
                self.optimizers[i].step()

    def get_weights(self):
        return {i: {k: v.cpu().clone() for k, v in self.models[i].state_dict().items()} for i in range(self.num_models)}
        
    def set_weights(self, weights_dict):
        for i in range(self.num_models):
            self.models[i].load_state_dict(weights_dict[i])


############################
# Main Driver
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "rcv1", 'ijcnn', 'w1a', 'a1a'])
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--model", type=str, default="fcnet_mnist")
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument('--noise-std', type=float, nargs='+', default=[0.0])
    
    # Local SGD hyperparameters
    parser.add_argument("--M", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--R", type=int, default=100, help="Number of communication rounds")
    parser.add_argument("--K", type=int, default=10, help="Number of local steps per round")
    
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every eval-interval total steps (e.g. R*K)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--n-pairs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu") 
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--log-root", type=str, default="Exp_LocalSGD")
    parser.add_argument("--save-prefix", type=str, default="")
    
    args = parser.parse_args()

    if ray.is_initialized():
        ray.shutdown()
        
    if args.device == 'cuda':
        ray.init(num_gpus=1, ignore_reinit_error=True, include_dashboard=False)
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    print(f"Ray initialized. Resources: {ray.cluster_resources()}")

    for r in range(1, args.repeats + 1):
        print(f"\n=== Repeat {r} ===")
        current_seed = args.seed_base + (r - 1) * 10
        args.seed_base = current_seed
        
        sub = args.save_prefix
        
        # Data preparation (per repeat)
        temp_data = load_data(args.dataset, args.dataset_path, 'train')
        total_req = args.num_samples
        neighbor_datasets = create_neibordataset(temp_data, total_req, args.n_pairs, seed=current_seed)
        
        for n_std in args.noise_std:
            print(f"[Repeat {r}/{args.repeats}] Seed={current_seed} | noise_std={n_std}")
            
            base_dir = os.path.join(args.log_root, f"{sub}{args.dataset}", f"nstd_{n_std}", f"r{r}")
            
            log_paths = {
                'log_dir': os.path.join(base_dir, "logs"),
                'rec_dir': os.path.join(base_dir, "records"),
                'ckpt_dir': os.path.join(base_dir, "checkpoints"),
                'log_name': f"log_localsgd_M{args.M}_R{args.R}_K{args.K}.txt",
                'csv_name': f"record_localsgd_M{args.M}_R{args.R}_K{args.K}_lr{args.lr}_nstd{n_std}.csv",
                'context': f"localsgd_M{args.M}_nstd{n_std}"
            }
            
            # Evaluate tools on Driver
            driver_device = args.device if torch.cuda.is_available() else 'cpu'
            num_models = args.n_pairs + 1
            driver_models = {i: load_model(args.model, args.loss, q=args.q).to(driver_device) for i in range(num_models)}
            
            test_dataset = load_data(args.dataset, args.dataset_path, 'test')
        
            used_indices = list(dict.fromkeys(neighbor_datasets[0]))
            train_sub_dataset = Subset(temp_data, used_indices)
            
            logger = Logger(log_paths['log_dir'], log_paths['log_name'])
            recorder = DataRecorder(log_paths['rec_dir'], log_paths['csv_name'])
            checkpoint = ModelCheckpoint(log_paths['ckpt_dir'], log_paths['context'])
            
            # Initialize Workers
            worker_actors = []
            for i in range(args.M):
                # All workers see the exact same subset of data for all neighbor models
                w_indices = neighbor_datasets
                res_opts = {"num_cpus": 0.2} 
                if args.device == 'cuda':
                    res_opts["num_gpus"] = 1.0 / (args.M + 1)
                elif args.device == 'cpu':
                    res_opts["num_cpus"] = max(0.2, 1.0 * ray.cluster_resources().get("CPU", 1) / (args.M + 1))
                
                w = LocalSGDWorker.options(**res_opts).remote(i, args, w_indices, n_std)
                worker_actors.append(w)
                
            print(f"Workers spawned. Starting {args.R} rounds, {args.K} max steps per round...")
            
            # Sync Initial Weights
            init_weights = {i: {k: v.cpu().clone() for k, v in driver_models[i].state_dict().items()} for i in range(num_models)}
            ray.get([w.set_weights.remote(init_weights) for w in worker_actors])
            
            total_steps = 0
            
            for rnd in range(1, args.R + 1):
                # 1. Local Training (Parallel over M workers for K steps locally)
                ray.get([w.train_k_steps.remote(args.K) for w in worker_actors])
                total_steps += args.K
                
                # 2. Gather & Average
                all_worker_weights = ray.get([w.get_weights.remote() for w in worker_actors])
                
                avg_weights = {}
                for m_idx in range(num_models):
                    avg_weights[m_idx] = {}
                    keys = all_worker_weights[0][m_idx].keys()
                    for key in keys:
                        stacked = torch.stack([worker_ws[m_idx][key].float() for worker_ws in all_worker_weights])
                        avg_weights[m_idx][key] = stacked.mean(dim=0).type_as(all_worker_weights[0][m_idx][key])
                
                # Broadcast Averaged Weights
                ray.get([w.set_weights.remote(avg_weights) for w in worker_actors])
                
                # Update driver models for evaluation
                for m_idx in range(num_models):
                    driver_models[m_idx].load_state_dict(avg_weights[m_idx])
                
                # 3. Evaluate
                check_points = [100, 200, 300, 400, 500, 600, 700, 800, 900]
                if total_steps % args.eval_interval == 0 or total_steps == 1 or total_steps in check_points or rnd == args.R:
                    
                    # Stability
                    diff_norm = 0
                    params0 = dict(driver_models[0].named_parameters())
                    for n in range(args.n_pairs):
                        paramsN = dict(driver_models[n+1].named_parameters())
                        local_diff = 0
                        for name in params0.keys():
                            local_diff += torch.sum((params0[name] - paramsN[name])**2).item()
                        diff_norm += np.sqrt(local_diff)
                    stability = diff_norm / args.n_pairs
                    
                    # Loss & Acc
                    tr_loss, tr_acc = run_inference(driver_models[0], train_sub_dataset, driver_device)
                    te_loss, te_acc = run_inference(driver_models[0], test_dataset, driver_device)
                    gap = te_loss - tr_loss
                    
                    print(f"Round:{rnd}| Step:{total_steps}| Stab:{stability:.6f}| TrLoss:{tr_loss:.6f}| TrAcc:{tr_acc:.6f}| TeLoss:{te_loss:.6f}| TeAcc:{te_acc:.6f}| Gap:{gap:.6f}")
                    logger.update(total_steps, tr_loss, tr_acc, te_loss, te_acc, gap, stability)
                    
                    rec_data = {
                        'round': rnd, 'step': total_steps,
                        'train_loss': tr_loss, 'train_acc': tr_acc,
                        'test_loss': te_loss, 'test_acc': te_acc,
                        'generalization_gap': gap, 'stability': stability
                    }
                    recorder.update(rec_data)
                    
                    if rnd == args.R:
                        recorder.save()
                        checkpoint.save(driver_models[0], total_steps)

            for w in worker_actors:
                ray.kill(w)
            del worker_actors

    ray.shutdown()