<h1 align="center">Community Detection</h1>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> Final project of DSC210 - Numerical Linear Algebra (2024 FALL)
    <br> 
</p>

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About ](#-about-)
- [📂 Datasets ](#-datasets-)
- [🎈 Usage ](#-usage-)
- [✨ Experiment Results ](#-experiment-results-)
  - [Metrics](#metrics)
  - [Visualization](#visualization)
- [✍️ Authors ](#️-authors-)
- [✨ Reference ](#-reference-)

## 🧐 About <a name = "about"></a>

This repository contains the final project for DSC210 - Numerical Linear Algebra. The objective of this project is to implement and evaluate two different community detection algorithms on the Cora dataset. The two algorithms are:

1. **Spectral Clustering**: Numerial linear algebra approach
2. **CDNMF: Contrastive Deep Nonnegative Matrix Factorization**: Deep learning approach

## 📂 Datasets <a name="datasets"></a>

The repository uses Cora as the dataset. The code takes input of graph structure by csv and txt files. Sample graphs for the Cora can 
be found in `dataset/cora`.

## 🎈 Usage <a name="usage"></a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hardness1020/Community_Detection/blob/main/main.ipynb)  

Click the badge above to run the code in Colab!  **Remember to use the GPU runtime for faster computation.**

The code is implemented in Python and designed to run in Jupyter Notebook and Colab. You can find the code in the main.ipynb file. It is modular and structured, allowing you to execute it step-by-step by running the cells in the notebook.


## ✨ Experiment Results <a name="experiment_results"></a>

The results might vary slightly due to the randomness of the algorithms. The following are the example results of the two algorithms on the Cora dataset.

### Metrics
The valuation metrics including:
- **Accuracy**: A ratio of the number of correctly classified samples to the total number of samples.
- **Normalized Mutual Information (NMI)**: A measure of the similarity between two clusterings of a set of data. It is used to assess the quality of clusters created by an algorithm against some ground truth.
- **Adjusted Rand Index (ARI)**: A measure of the similarity between two data clusterings. It considers all pairs of samples and counts pairs that are assigned in the same or different clusters in the predicted and true clusterings.

![image](https://github.com/hardness1020/Community_Detection/blob/main/fig/matrics_table.png?raw=true)

### Visualization
The following are data visualizations of predicted communities generated by the two approaches. In these visualizations, nodes represent papers, edges represent citations, and the colors of nodes represent which category the paper belongs.

![image](https://github.com/hardness1020/Community_Detection/blob/main/fig/visualization_comparison.png?raw=true)


## ✍️ Authors <a name="authors"></a>
- [@hardness1020](https://github.com/hardness1020) - Marcus Chang
- [@ripple-space](https://github.com/ripple-space) - Yi Lien
- [@Jeanhsu0707](https://github.com/Jeanhsu0707) - Hua Jin Hsu
- [@TingShiuanLai](https://github.com/TingShiuanLai) - Ting-Shiuan, Lai


## ✨ Reference <a name="reference"></a>
- [Cora Dataset](https://ieee-dataport.org/documents/cora)
- [CDNMF: Contrastive Deep Nonnegative Matrix Factorization](https://github.com/6lyc/CDNMF)
- [Spectral Clustering](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.SpectralClustering.html)
