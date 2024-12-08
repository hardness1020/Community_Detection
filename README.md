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
- [📖 Requirement ](#-requirement-)
- [📂 Datasets ](#-datasets-)
- [🎈 Usage ](#-usage-)
- [✨ Experiment Results ](#-experiment-results-)
- [✍️ Authors ](#-authors-)

## 🧐 About <a name = "about"></a>

This repository contains the final project for DSC210 - Numerical Linear Algebra. The objective of this project is to implement and evaluate various community detection algorithms, including:

1. Spectral Clustering
2. CDNMF: Contrastive Deep Nonnegative Matrix Factorization

## 📖 Requirement <a name="requirement"></a>

To install the dependencies, please follow the colab instruction to use: `%pip install -r requirements.txt`.

## 📂 Datasets <a name="datasets"></a>

The code takes input graphs in csv and txt files. Sample graphs for the Cora could also be found in the repository: `dataset/cora`.

## 🎈 Usage <a name="usage"></a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hardness1020/Community_Detection/blob/main/main.ipynb)
Click the badge above to run the code in Colab!

The code is implemented in Python and designed to run in Jupyter Notebook and Colab. You can find the code in the main.ipynb file. It is modular and structured, allowing you to execute it step-by-step by running the cells in the notebook.


## ✨ Experiment Results <a name="experiment_results"></a>
The following are data visualizations of predicted communities generated by the two approaches. In these visualizations:
* Data points with the same color represent the same predicted community.
* Ground-truth communities are also provided at the bottom for comparison.
Feel free to explore and analyze how the predicted communities align with the ground truth.

1. Spectral Clustering
![image](https://github.com/user-attachments/assets/852d5491-fd96-4c0f-bbd2-3325d6daf402)

2. Contrastive Deep Nonnegative Matrix Factorization
![image](https://github.com/user-attachments/assets/56300ca0-5a71-4218-ad64-45ed2d93322e)

3. Comparison with Groud-truth
![image](https://github.com/user-attachments/assets/8e4e33c2-e885-464c-b4f2-0b0110078b3f)

<!-- ## ⛏️ Built Using <a name = "built_using"></a>

- [??]() - Database
- [Communities](https://github.com/shobrook/communities?tab=readme-ov-file) - Package of Methods for Community Detection
- [CDNMF](https://github.com/6lyc/CDNMF?tab=readme-ov-file) - Method for Community Detection -->


## ✍️ Authors <a name="authors"></a>
- [@hardness1020](https://github.com/hardness1020) - Marcus Chang
- [@ripple-space](https://github.com/ripple-space) - Yi Lien
- [@Jeanhsu0707](https://github.com/Jeanhsu0707) - Hua Jin Hsu
- [@TingShiuanLai](https://github.com/TingShiuanLai) - Ting-Shiuan, Lai

<!-- 
## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- Hat tip to anyone whose code was used
- Inspiration
- References -->
