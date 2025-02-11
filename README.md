# BiLB4MTL
Official implementation of *Scalable Bilevel Loss Balancing for Multi-Task Learning*.

Multi-task learning (MTL) has been widely adopted for its ability to simultaneously learn multiple tasks. While existing gradient manipulation methods often yield more balanced solutions than simple scalarization-based approaches, they typically incur a significant computational overhead of 
$\mathcal{O}(K)$ in both time and memory, where $K$is the number of tasks. In this paper, we propose BiLB4MTL, a simple and scalable loss balancing approach for MTL, formulated from a novel bilevel optimization perspective. Our method incorporates three key components: (i) an initial loss normalization, (ii) a bilevel loss-balancing formulation, and (iii) a scalable first-order algorithm that requires only $\mathcal{O}(1)$ time and memory. Theoretically, we prove that BiLB4MTL guarantees convergence not only to a stationary point of the bilevel loss balancing problem but also to an $\epsilon$-accurate Pareto stationary point for all $K$ loss functions under mild conditions. Extensive experiments on diverse multi-task datasets demonstrate that BiLB4MTL achieves state-of-the-art performance in both accuracy and efficiency.


### Setup Environment
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
