# Collaboratively Learning Federated Models from Noisy Decentralized Data
<div align="center">

<div>
    <a href='https://www.linkedin.com/in/haoyuan-li-cs9654/' target='_blank'>Haoyuan Li</a><sup>1</sup>&emsp;
    <a href='https://mathias-funk.com/' target='_blank'>Mathias Funk</a><sup>1</sup>&emsp;
    <a href='https://nezihemervegurel.github.io/' target='_blank'>Nezihe Merve Gürel</a><sup>2</sup>&emsp;
    <a href='https://aqibsaeed.github.io/' target='_blank'>Aaqib Saeed</a><sup>1</sup>&emsp;
</div>
<div>
<sup>1</sup><a href="https://www.tue.nl/en/our-university/departments/industrial-design/research/our-research-labs/decentralized-artificial-intelligence-research-lab" target="_blank" rel="noopener noreferrer">
                        Decentralized Artificial Intelligence Research Lab, Eindhoven University of Technology
                      </a>&emsp;

<sup>2</sup>Delft University of Technology&emsp;
</div>
</div>


<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Decentralized-AI-Reserach-Lab.FedNS&left_color=blue&right_color=green)
[![arXiv](https://img.shields.io/badge/arXiv-2409.02189-blue?logo=arxiv&logoColor=orange)](https://arxiv.org/abs/2409.02189)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Citations-purple.svg)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Collaboratively+Learning+Federated+Models+from+Noisy+Decentralized+Data&btnG=)
[![Project Page](https://img.shields.io/badge/Project%20Page-Online-brightgreen)](https://haoyuan-l.github.io/fedns/)
[![IEEE BigData 2024](https://img.shields.io/badge/IEEE%20BigData%202024-Accepted-yellow.svg)](https://www3.cs.stonybrook.edu/~ieeebigdata2024/index.html)

</div>


## 📢 Updates
[11/2024] Released the project page [link](https://haoyuan-l.github.io/fedns/).

[10/2024] Code has been released.

[10/2024] Accepted to IEEE BigData 2024

[09/2024] [arXiv](http://www.arxiv.org/abs/2409.02189) paper has been released.


## 📝 Abstract
Federated learning (FL) has emerged as a prominent method for collaboratively training machine learning models using local data from edge devices, all while keeping data decentralized. However, accounting for the quality of data contributed by local clients remains a critical challenge in FL, as local data are often susceptible to corruption by various forms of noise and perturbations, which compromise the aggregation process and lead to a subpar global model. In this work, we focus on addressing the problem of noisy data in the input space, an under-explored area compared to the label noise. We propose a comprehensive assessment of client input in the gradient space, inspired by the distinct disparity observed between the density of gradient norm distributions of models trained on noisy and clean input data. Based on this observation, we introduce a straightforward yet effective approach to identify clients with low-quality data at the initial stage of FL. Furthermore, we propose a noise-aware FL aggregation method, namely **Fed**erated **N**oise-**S**ifting (**FedNS**), which can be used as a plug-in approach in conjunction with widely used FL strategies. Our extensive evaluation on diverse benchmark datasets under different federated settings demonstrates the efficacy of **FedNS**. Our method effortlessly integrates with existing FL strategies, enhancing the global model’s performance by up to 13.68% in IID and 15.85% in non-IID settings when learning from noisy decentralized data.

<div align="center">
<h3>⚓ Overview ⚓</h3>
<img src="img/overview.png" width="95%">
<h3>📚 Three Pillars of FedNS</h3>
</div>

- **🔍 Noise Identification:** FedNS identifies noisy clients in the first training round (one-shot).
- **🛡️ Resilient Aggregation:** A resilient strategy that minimizes the impact of noisy clients, ensuring robust model performance.
- **🔒 Data Confidentiality:** Shares only scalar gradient norms to keep data confidential.

## 🔧 Requirements
###  Environment 
1. [torch 2.2.0+cu118](https://github.com/pytorch/pytorch)
2. [torchvision 0.17.0+cu118](https://github.com/pytorch/vision)
3. [numpy 1.21.6](https://github.com/numpy/numpy.git)
3. [flwr 0.6.12](https://github.com/adap/flower)

### Dataset 
We provide the noisy dataset creation on various benchmarks, we shown an example of generating noisy CIFAR10:
- **CIFAR10**:
```
cd ./data/cifar10data
python create_cifar10_noisy.py
```
For benchmark on human annoation errors, you can refer to [CIFAR10/100N](https://github.com/UCSC-REAL/cifar-10-100n). For decentralized data generation, please go to folder [.\src_fed](https://github.com/Decentralized-AI-Reserach-Lab/FedNS/tree/main/src_fed/cifar10).

## 💡 Running scripts

To prepare your experiment, please setup your configuration at the main.py. You can configure the specific federated learning strategy at server.py. You can simply execute the main script them to run the experiment, the results will save as a `logs` file.

```
cd ./scr_fed/cifar10
python main.py
```

## 💭 Correspondence
If you have any questions, please contact me via [email](h.y.li@tue.nl) or open an [issue](https://github.com/Decentralized-AI-Reserach-Lab/FedNS/issues).

## Citing FedNS
The code repository for "[Collaboratively Learning Federated Models from Noisy Decentralized Data](https://arxiv.org/abs/2409.02189)" (IEEE BigData 2024) in PyTorch.  If you use any content of this repo for your work, please cite the following bib entry: 

```bibtex
@inproceedings{li2024collaboratively,
  title={Collaboratively Learning Federated Models from Noisy Decentralized Data},
  author={Li, Haoyuan and Funk, Mathias and G{\"u}rel, Nezihe Merve and Saeed, Aaqib},
  booktitle={2024 IEEE International Conference on Big Data (BigData)},
  pages={7879--7888},
  year={2024},
  organization={IEEE}
}
```
