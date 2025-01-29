<div align="center">

<img src="https://raw.githubusercontent.com/yutanagano/sceptr/main/sceptr.svg" width=700>

[![Latest release](https://img.shields.io/pypi/v/sceptr)](https://pypi.org/p/sceptr)
![Tests](https://github.com/yutanagano/sceptr/actions/workflows/tests.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/sceptr/badge/?version=latest)](https://sceptr.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/yutanagano/sceptr?tab=MIT-1-ov-file#readme)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.cels.2024.12.006-pink)](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00369-7)

### Check out the [documentation page](https://sceptr.readthedocs.io).

</div>

---

| <img src="https://raw.githubusercontent.com/yutanagano/sceptr/main/docs/graphical_abstract.png" width=700> |
|-|
| Graphical abstract. Traditional protein language models that are trained purely on masked-language modelling underperform sequence alignment models on TCR specificity prediction. In contrast, our model SCEPTR is jointly trained on masked-language modelling and contrastive learning, allowing it to outperform other language models as well as the best sequence alignment models to achieve state-of-the-art performance. |

**SCEPTR** (**S**imple **C**ontrastive **E**mbedding of the **P**rimary sequence of **T** cell **R**eceptors) is a small, fast, and informative TCR representation model that can be used for alignment-free TCR  analysis, including for TCR-pMHC interaction prediction and TCR clustering (metaclonotype discovery).
Our [manuscript](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00369-7) demonstrates that SCEPTR can be used for few-shot TCR specificity prediction with improved accuracy over previous methods.

SCEPTR is a BERT-like transformer-based neural network implemented in [Pytorch](https://pytorch.org).
With the default model providing best-in-class performance with only 153,108 parameters (typical protein language models have tens or hundreds of millions), SCEPTR runs fast- even on a CPU!
And if your computer does have a [CUDA-enabled GPU](https://en.wikipedia.org/wiki/CUDA), the sceptr package will automatically detect and use it, giving you blazingly fast performance without the hassle.

sceptr's API exposes four intuitive functions: `calc_cdist_matrix`, `calc_pdist_vector`, `calc_vector_representations`, and `calc_residue_representations` -- and it's all you need to make full use of the SCEPTR models.
What's even better is that they are fully compliant with [pyrepseq](https://pyrepseq.readthedocs.io)'s [tcr_metric](https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.metric.tcr_metric.TcrMetric) API, so sceptr will fit snugly into the rest of your repertoire analysis workflow.

## Installation

```bash
pip install sceptr
```

## Citing SCEPTR
Please cite our [manuscript](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00369-7).

### BibTex
```bibtex
@article{nagano_contrastive_2025,
	title = {Contrastive learning of {T} cell receptor representations},
	volume = {16},
	issn = {2405-4712, 2405-4720},
	url = {https://www.cell.com/cell-systems/abstract/S2405-4712(24)00369-7},
	doi = {10.1016/j.cels.2024.12.006},
	language = {English},
	number = {1},
	urldate = {2025-01-19},
	journal = {Cell Systems},
	author = {Nagano, Yuta and Pyo, Andrew G. T. and Milighetti, Martina and Henderson, James and Shawe-Taylor, John and Chain, Benny and Tiffeau-Mayer, Andreas},
	month = jan,
	year = {2025},
	pmid = {39778580},
	note = {Publisher: Elsevier},
	keywords = {contrastive learning, protein language models, representation learning, T cell receptor, T cell specificity, TCR, TCR repertoire},
}
```
