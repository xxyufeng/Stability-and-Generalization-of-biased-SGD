import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, Subset
from torch.func import functional_call, vmap, grad
import numpy as np
import argparse
import pandas as pd
import time

from dataset_and_model import load_data, load_model

def create_neibordataset(dataset, total_samples, n_pairs, seed=None):
    if seed is not None:
        np.random.seed(42+seed)
    indices = list(range(len(dataset)))
    sample_count = min(len(dataset), total_samples+n_pairs)
    indices = np.random.choice(indices, sample_count, replace=False)
    
    print('----Creating neighboring dataset----')
    print(f'Total used samples: {len(indices)}')

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
            pass
    return datasets_dict

class Logger:
    def __init__(self, log_dir, log_filename):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_filename)

    def update(self, iteration, t_loss, t_acc, te_loss, te_acc, gap, stab):
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(f"Iter: {iteration}, "
                    f"TrLoss: {t_loss:.4f}, TrAcc: {t_acc:.4f}, "
                    f"TeLoss: {te_loss:.4f}, TeAcc: {te_acc:.4f}, "
                    f"Gap: {gap:.4f}, Stab: {stab:.6f}")
            f.write('\n')

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

class SGDExperiment:
    def __init__(self, args, datasets_dict, log_paths, current_noise_std=0.0, current_seed=42):
        self.args = args
        self.neighbor_datasets_dict = datasets_dict
        self.dataset_len = len(self.neighbor_datasets_dict[0])
        
        self.current_noise_std = current_noise_std
        self.current_seed = current_seed
        self.iteration = 1
        self.logger = Logger(log_paths['log_dir'], log_paths['log_name'])
        self.recorder = DataRecorder(log_paths['rec_dir'], log_paths['csv_name'])
        self.checkpoint = ModelCheckpoint(log_paths['ckpt_dir'], log_paths['context'])
        
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        self.num_models = args.n_pairs + 1
        self.alg = args.alg
        self.clip_norm = args.clip_norm
        self.mu = args.zero_mu
        self.K = args.zero_K
        self.random_k = args.random_k
        self.models = {i: load_model(args.model, args.loss, q=args.q).to(self.device) for i in range(self.num_models)}
        self.optimizers = {i: optim.SGD(self.models[i].parameters(), lr=args.lr) for i in range(self.num_models)}
        
        self.full_train = load_data(args.dataset, args.dataset_path, 'train')
        self.test_dataset = load_data(args.dataset, args.dataset_path, 'test')

        ## Data preparation:
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

        used_indices = list(dict.fromkeys(self.neighbor_datasets_dict[0]))
        self.train_sub_dataset = Subset(self.full_train, used_indices)
        self.perm_indices = np.arange(self.dataset_len)
        self.curr_pos = 0
            
        torch.manual_seed(args.seed_base)
        np.random.seed(args.seed_base)
        np.random.shuffle(self.perm_indices)

    def _get_next_batch_indices(self):
        if self.curr_pos >= self.dataset_len:
            np.random.shuffle(self.perm_indices)
            self.curr_pos = 0
            
        end_pos = min(self.curr_pos + self.args.batch_size, self.dataset_len)
        batch_positions = self.perm_indices[self.curr_pos : end_pos]
        self.curr_pos = end_pos
        return batch_positions

    def _fetch_data(self, model_idx, batch_positions):
        real_indices = [self.neighbor_datasets_dict[model_idx][p] for p in batch_positions]
        if len(real_indices) > 0:
            inputs = self.cached_x[real_indices]
            labels = self.cached_y[real_indices]
        else:
            inputs = torch.empty(0)
            labels = torch.empty(0)
        return inputs, labels
    
    def sample_from_unit_sphere(self, shape, random_seed=42):
        """
        Sample uniformly from the unit sphere
        Args:
            shape: Shape of the sampling vector
            device: Device (cpu/gpu)
        Returns:
            u: Random vector on the unit sphere (L2 norm = 1)
        """
        # Generate Gaussian random vector
        u = torch.randn(shape, device=self.device, generator=torch.Generator(device=self.device).manual_seed(random_seed))
        # Calculate L2 norm
        norm = torch.norm(u.view(-1), p=2)
        # Normalize to unit sphere
        if norm > 1e-10:  # Avoid division by zero
            u = u / norm
        else:
            # Resample if norm is too small
            u = torch.randn(shape, device=self.device, generator=torch.Generator(device=self.device).manual_seed(random_seed))
            norm = torch.norm(u.view(-1), p=2)
            u = u / norm
        return u
    
    def approx_gradient_sphere(self, model, inputs, labels, original_loss, random_seed=42):
        """
        Approximate gradient of model parameters (Zeroth-order gradient estimation) - using unit sphere sampling
        Args:
            model: Model for gradient estimation (generator or discriminator)
            loss_fn: Loss function (used to calculate loss after perturbation)
            inputs: Input data required for loss calculation (varies by model type)
            labels: Target labels for loss calculation
            original_loss: Loss under original parameters
            mu: Perturbation step size (μ1 or μ2)
            K: Number of sampling iterations (used to reduce variance by averaging)
            device: Device (cpu/gpu)
        Returns:
            approx_grads: List of approximate gradients matching the model parameter structure
        """
        # Initialize approximate gradients (same structure as model parameters)
        approx_grads = [torch.zeros_like(param) for param in model.parameters()]
        model.eval()  # Set model to evaluation mode for consistent loss calculation
        
        with torch.no_grad():
            for i in range(self.K):
                u_list = []
                # 1. Sample from unit sphere for each parameter
                seed = random_seed + i  # Different seed for each sampling iteration
                for param in model.parameters():
                    u = self.sample_from_unit_sphere(param.shape, seed)
                    u_list.append(u)
                
                # 2. Apply perturbation to model parameters: w + μ*u
                for param, u in zip(model.parameters(), u_list):
                    param.data.add_(self.mu * u)  # Temporarily modify parameters
                
                # 3. Calculate loss after perturbation
                _, perturbed_loss = model(inputs, labels)
                
                # 4. Restore original parameters (avoid perturbation affecting subsequent calculations)
                for param, u in zip(model.parameters(), u_list):
                    param.data.sub_(self.mu * u)  # Restore parameters
                
                # 5. Accumulate gradient approximation: (f(w+μu) - f(w))/μ * u * d
                # where d is the parameter dimension for unbiased estimation
                loss_diff = (perturbed_loss - original_loss) / self.mu
                for i, (u, grad) in enumerate(zip(u_list, approx_grads)):
                    # Multiply by parameter dimension to get unbiased estimation
                    param_dim = u.numel()
                    approx_grads[i] += loss_diff * u * param_dim
            
        # 6. Average results over K sampling iterations
        for grad in approx_grads:
            grad.div_(self.K)

        model.train()  # Restore model to training mode
        
        return approx_grads
    
    def approx_gradient_random_k(self, model, inputs, labels, random_k, random_seed=42):
        """
        For per-sample gradients, perform global Random-k sparsification, then take the average
        """
        # ====== 1. Get per-sample gradients ======
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def compute_single_loss(params, buffers, x, y):
            # Compute loss for a single sample (x, y) using functional_call to avoid modifying model parameters
            out, loss = functional_call(model, (params, buffers), (x.unsqueeze(0), y.unsqueeze(0)))
            return loss

        # per_sample_grads_dict: [Batch_size, param_shape...]
        per_sample_grads_dict = vmap(grad(compute_single_loss), in_dims=(None, None, 0, 0))(params, buffers, inputs, labels)

        N = inputs.size(0)

        # ====== 2. Flatten and concatenate per-sample gradients ======
        flat_ps_grads_list = []
        for g in per_sample_grads_dict.values():
            flat_ps_grads_list.append(g.reshape(N, -1))
        
        # Shape: [Batch_size, total_param_dim]
        ps_grads_global_flat = torch.cat(flat_ps_grads_list, dim=1) 
        D_total = ps_grads_global_flat.size(1)
        
        # make sure random_k does not exceed total dimension
        actual_k = min(random_k, D_total)

        # ====== 3. Construct Random-K Mask ======
        # Core technique: Generate random noise [N, D_total], find indices of the largest actual_k elements in each row
        noise = torch.rand(N, D_total, device=self.device, generator=torch.Generator(device=self.device).manual_seed(random_seed))
        _, topk_idx = noise.topk(actual_k, dim=1)
        
        # Construct a zero matrix and set the positions corresponding to topk_idx to 1
        mask = torch.zeros(N, D_total, device=self.device)
        mask.scatter_(1, topk_idx, 1.0)


        # ====== 4. Mask gradients and average ======
        batch_grad_global = (ps_grads_global_flat * mask).mean(dim=0)

        # ====== 5. Restore gradient shape ======
        approx_grads = []
        offset = 0
        for param_name, param in model.named_parameters():
            numel = param.numel()
            grad_slice = batch_grad_global[offset : offset + numel]
            approx_grads.append(grad_slice.view_as(param))
            offset += numel

        return approx_grads

    def _run_inference(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4 if self.device == 'cuda' else 0) 
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, loss = model(x, y)
                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        model.train()
        return total_loss / total, correct / total

    def _evaluate_routine(self, current_iter):
        diff_norm = 0
        params0 = dict(self.models[0].named_parameters())
        for n in range(self.args.n_pairs):
            paramsN = dict(self.models[n+1].named_parameters())
            local_diff = 0
            for name in params0.keys():
                p0 = params0[name]
                pN = paramsN[name]
                local_diff += torch.sum((p0 - pN)**2).item()
            diff_norm += np.sqrt(local_diff)
        stability = diff_norm / self.args.n_pairs
        
        tr_loss, tr_acc = self._run_inference(self.models[0], self.train_sub_dataset)
        te_loss, te_acc = self._run_inference(self.models[0], self.test_dataset)

        gen_gap = te_loss - tr_loss

        print(f"Iter:{current_iter}| Stab:{stability:.6f}| Train Loss:{tr_loss:.6f}| Train Acc:{tr_acc:.6f}| Test Loss:{te_loss:.6f}| Test Acc:{te_acc:.6f}| Gap:{gen_gap:.6f}")

        self.logger.update(current_iter, tr_loss, tr_acc, te_loss, te_acc, gen_gap, stability)
        
        rec_data = {
            'iteration': current_iter,
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'test_loss': te_loss, 'test_acc': te_acc,
            'generalization_gap': gen_gap, 'stability': stability
        }
        self.recorder.update(rec_data)

        if current_iter >= self.args.iterations:
            self.recorder.save()
            # self.checkpoint.save(self.models[0], current_iter)

    def train_loop(self):
        check_points = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        
        while self.iteration <= self.args.iterations:
            batch_positions = self._get_next_batch_indices()
            
            for i in range(self.num_models):
                self.models[i].train()
                self.optimizers[i].zero_grad()
                
                inputs, labels = self._fetch_data(i, batch_positions)
                if inputs.size(0) == 0: continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.alg in ['zero_sgd', 'rk_sgd']:
                    with torch.no_grad():
                        outputs, loss = self.models[i](inputs, labels)
                else:
                    outputs, loss = self.models[i](inputs, labels)

                if self.alg == 'zero_sgd':
                    original_loss = loss.item()
                    seed = self.args.seed_base + self.iteration + 19
                    approx_grads = self.approx_gradient_sphere(self.models[i], inputs, labels, original_loss, random_seed=seed)
                    for param, approx_grad in zip(self.models[i].parameters(), approx_grads):
                        param.grad = approx_grad
                elif self.alg == 'clip_sgd':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), max_norm=self.clip_norm)
                elif self.alg == 'rk_sgd':
                    seed = self.args.seed_base + self.iteration + 19
                    approx_grads = self.approx_gradient_random_k(self.models[i], inputs, labels, random_k=self.random_k, random_seed=seed)
                    for param, approx_grad in zip(self.models[i].parameters(), approx_grads):
                        param.grad = approx_grad
                else:
                    loss.backward()
                
                self.optimizers[i].step()

            if self.iteration % self.args.eval_interval == 0 or self.iteration == 1 or self.iteration in check_points:
                self._evaluate_routine(self.iteration)
                
            self.iteration += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'rcv1', 'ijcnn', 'w1a', 'a1a'])
    parser.add_argument('--dataset-path', type=str, default='./data')
    parser.add_argument('--model', type=str, default='fcnet_mnist')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--alg', type=str, default='sgd', choices=['sgd', 'clip_sgd', 'zero_sgd', 'rk_sgd', 'local_sgd'])
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--zero-mu', type=float, default=0.01)
    parser.add_argument('--zero-K', type=int, default=10)
    parser.add_argument('--random-k', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--q', type=float, default=0.0)
    parser.add_argument('--noise-std', type=float, nargs='+', default=[0.0])
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--n-pairs', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu') 
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--log-root', type=str, default='Exp_SGD')
    parser.add_argument('--save-prefix', type=str, default='')
    
    args = parser.parse_args()

    for r in range(1, args.repeats + 1):
        print(f'\n=== Repeat {r} ===')
        current_seed = args.seed_base + (r - 1) * 10
        args.seed_base = current_seed

        sub = args.save_prefix
        
        # Load and sample dataset once per repetition inside the loop
        temp_data = load_data(args.dataset, args.dataset_path, 'train')
        total_req = args.num_samples
        neighbor_datasets = create_neibordataset(temp_data, total_req, args.n_pairs, seed=current_seed)
        del temp_data
        
        for n_std in args.noise_std:
            print(f'[Repeat {r}/{args.repeats}] Seed={current_seed} | noise_std={n_std}')
            
            base_dir = os.path.join(args.log_root, f'{sub}{args.dataset}', f'nstd_{n_std}', f'r{r}')
            
            log_paths = {
                'log_dir': os.path.join(base_dir, 'logs'),
                'rec_dir': os.path.join(base_dir, 'records'),
                'ckpt_dir': os.path.join(base_dir, 'checkpoints'),
                'log_name': f'log_{args.alg}.txt',
                'csv_name': f'record_{args.alg}_{args.iterations}iter_{args.lr}lr_nstd{n_std}.csv',
                'context': f'{args.alg}_nstd{n_std}'
            }
            
            experiment = SGDExperiment(args, neighbor_datasets, log_paths, current_noise_std=n_std, current_seed=current_seed)
            print(f'Training begins for alg={args.alg}, noise_std={n_std}...')
            experiment.train_loop()
            
        print(f'Repeat {r} finished.')
