# Stability-and-Generalization-of-biased-SGD
This repository is the experimental implementation of the section "Experimental Results" of paper "Optimistic Generalization Bounds of Biased Stochastic Gradient Methods"

---

## 📂 File Descriptions

### 1. `BiasSGD.py` (Stability Analysis of SGD, clip SGD, random-k SGD, zeroth order SGD)
This script focuses on analyzing the **Algorithmic Stability and Excess Risk** of some biased SGD variants. 
* **Mechanism:** It trains multiple models concurrently (n-pairs + 1 models by default) on neighboring datasets (datasets differing by exactly one data point). 
* **Key Metric:** Generalization Stability (measured as the L2 distance between the parameters of the reference model and neighboring models).

### 2. `Local SGD.py` (Stability Analysis of Local SGD using Ray package)
This script focuses on analyzing the **Algorithmic Stability and Excess Risk** of Local SGD.
* **Mechanism:** Using Ray package to implement distributed training with $M$ workers 

### 3. `dataset_and_model.py` (Data & Model Utilities)
This script serves as the core utility module for all data loading, preprocessing, and model architecture definitions used across the ASGD experiments.

*   **Datasets Supported:** 
    *   **Vision Datasets:** MNIST, CIFAR-10.
    *   **LibSVM Datasets (Sparse/Tabular):** RCV1, GISETTE, a1a, w1a, ijcnn.
*   **Models Supported:**
    *   **Linear/Convex Models:** Linear Classifiers for RCV1, GISETTE, MNIST, CIFAR-10, a1a, w1a, ijcnn (`Linear_RCV1`, `Linear_MNIST`, etc.).
    *   **Non-Convex Neural Networks:** Multilayer Perceptrons (`FCNET_MNIST`), and small Deep CNNs adapted for CIFAR-10 (`ResNet18_CIFAR10`, `MobileNetV1_CIFAR10`, `ShuffleNetV2_CIFAR10`, `ResNet20_CIFAR10`).
    *   **Loss Functions:** Wraps training targets automatically using Mean Squared Error (`mse`), Cross Entropy (`ce`), or parameterized Hinge Loss (`hingeloss`).
 
### 4.  `visualize_stability_evolution.py` and `visualize_test_error_evolution.py`
This script focuses on drawing the evolution of stability and test error.

---

## ⚙️ Core Arguments

When executing the baseline scripts (`BiasSGD.py` or `Local SGD.py`), the following arguments dictate the experiment's behavior:

### Data & Model Configuration
*   `--dataset`: Name of the dataset to run (e.g., `mnist`,`rcv1`).
*   `--dataset-path`: Destination path for downloading and parsing datasets. Default is `./data`.
*   `--model`: Name of the model mapping to `dataset_and_model.py` string literals (e.g., `fcnet_mnist`, `resnet20_cifar10`, `linear_rcv1`).
*   `--loss`: Loss function choice. Options: `mse`, `ce`, `hingeloss`.
*   `--q`: Parameter for q-norm hinge loss ( (q-1,L)-Holder continous, $\forall q\in [1,2]$ ). if `hingeloss` is selected (e.g., `q=1.5` for parameterized hinge loss). 

### Algorithm Setup
*   `--lr`: Base Learning rate for the SGD optimizer.
*   `--alg`: Algorithm with choices in ['sgd', 'clip_sgd', 'zero_sgd', 'rk_sgd', 'local_sgd']
*   `--batch-size`: Mini-batch size processed by each worker locally (Default is often `1` to simulate pure stochastic algorithms).
*   `--iterations`: Total number of global parameter updates the server actor will perform before terminating.
*   `--repeats`: Number of times the entire experiment runs with different random seeds.

#### Clip SGD
*   `--clip-norm`: Clip norm of Clip SGD.

#### Zeroth order SGD
*   `--zero-mu`: smoothing parameter in approximating the gradient of f.
*   `--zero-K`: number of random directions to reduce the variance in the gradient estimation.

#### Zeroth order SGD
*   `--random-k`: number of indices used for gradients update.


### Others
*   `--noise-std`: variance of gaussian noise added to the inputs.
*   `--n-pairs`: Controls $N$, the number of neighboring substituted datasets analyzed simultaneously for theoretical algorithmic stability evaluations. 

---

## 🚀 How to Run

**1. Install Dependencies:**

!pip install torch ray numpy pandas matplotlib

**2. Running the BiasSGD Experiment:**

!python BiasSGD.py --dataset mnist --dataset-path \experiment\data --model linear_mnist --loss ce --alg zero_sgd --zero-mu 0.00001 --zero-K 100 --lr 0.001 --noise-std 0 0.5 1 1.5 --iterations 20000 --eval-interval 2000 --repeats 5 --num-samples 10000 --batch-size 8 --log-root Exp\zeroSGD

or

!python BiasSGD.py --dataset mnist --dataset-path \experiment\data --model linear_mnist --loss ce --alg rk_sgd --random-k 500 --lr 0.001 --noise-std 0 0.5 1 1.5 --iterations 20000 --eval-interval 2000 --repeats 5 --num-samples 10000 --batch-size 8 --log-root Exp\rkSGD   

**3. Running the Local SGD Experiment:**

!python LocalSGD_noise.py --dataset mnist --model linear_mnist --lr 0.001 --M 8 --R 200 --K 100 --noise-std 0 0.5 1 1.5 --eval-interval 2000 --num-samples 10000 --device cpu --repeats 5
