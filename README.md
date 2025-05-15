# LDC-MTL
Official implementation of *LDC-MTL: Balancing Multi-Task Learning through Scalable Loss Discrepancy Control*. [Paper link](https://www.arxiv.org/abs/2502.08585)

Multi-task learning (MTL) has been widely adopted for its ability to simultaneously learn multiple tasks. While existing gradient manipulation methods often yield more balanced solutions than simple scalarization-based approaches, they typically incur a significant computational overhead of $\mathcal{O}(K)$ in both time and memory, where $K$ is the number of tasks. In this paper, we propose LDC-MTL, a simple and scalable loss discrepancy control approach for MTL, formulated from a bilevel optimization perspective. Our method incorporates three key components: (i) a coarse loss pre-normalization, (ii) a bilevel formulation for fine-grained loss discrepancy control, and (iii) a scalable first-order bilevel algorithm that requires only $\mathcal{O}(1)$ time and memory. Theoretically, we prove that LDC-MTL guarantees convergence not only to a stationary point of the bilevel problem with loss discrepancy control but also to an $\epsilon$-accurate Pareto stationary point for all $K$ loss functions under mild conditions. Extensive experiments on diverse multi-task datasets demonstrate the superior performance of LDC-MTL in both accuracy and efficiency.

---

<p align="center"> 
    <img src="https://github.com/OptMN-Lab/-BiLB4MTL/blob/main/figs/flowchart.png" width="800">
</p>

Our bilevel loss balancing pipeline for multi-task learning. First, task losses will be normalized through an initial loss normalization module. Then, the lower-level problem optimizes the model parameter $x^t$ by minimizing the weighted sum of task losses and the upper-level problem optimizes the router model parameter $W^t$ for task balancing.

---

<p align="center"> 
    <img src="https://github.com/OptMN-Lab/-BiLB4MTL/blob/main/figs/toy_example.png" width="800">
</p>

The loss trajectories of a toy 2-task learning problem and the runtime comparison of different MTL methods for 50000 steps. ★ on the Pareto front denotes the converge points. Although FAMO achieves more balanced results than LS and MGDA, it converges to different points on the Pareto front. Our method reaches the same balanced point with a computational cost comparable to the simple Linear Scalarization (LS). 

---

<p align="center"> 
    <img src="https://github.com/OptMN-Lab/-BiLB4MTL/blob/main/figs/trainingtime_comparison_4datasets.png" width="800">
</p>

Time scale comparison among well-performing approaches, with LS considered the reference method for standard time.

---

## Datasets
The performance is evaluated under 4 datasets:
 - Image-level Classification. The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 40 tasks.
 - Regression. The QM9 dataset contains 11 tasks, which can be downloaded automatically from Pytorch Geometric.
 - Dense Prediction. The [NYU-v2](https://github.com/lorenmt/mtan) dataset contains 3 tasks and the [Cityscapes](https://github.com/lorenmt/mtan) dataset (UPDATE: the small version) contains 2 tasks.


## Setup Environment
Create the environment:
```
conda create -n mtl python=3.9.7
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
```
Then, install the repo:
```
https://github.com/OptMN-Lab/-BiLB4MTL.git
cd -BiLB4MTL
python -m pip install -e .
```

