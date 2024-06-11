<div align="center">

# SCEPTR

[![Latest release](https://img.shields.io/pypi/v/sceptr)](https://pypi.org/p/sceptr)
![Tests](https://github.com/yutanagano/sceptr/actions/workflows/tests.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/sceptr/badge/?version=latest)](https://sceptr.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/yutanagano/sceptr?tab=MIT-1-ov-file#readme)

### Check out the [documentation page](https://sceptr.readthedocs.io).

</div>

**SCEPTR** (**S**imple **C**ontrastive **E**mbedding of the **P**rimary sequence of **T** cell **R**eceptors) is a small, fast, and accurate TCR representation model that can be used for alignment-free TCR  analysis, including for TCR-pMHC interaction prediction and TCR clustering (metaclonotype discovery).
Our [preprint](https://arxiv.org/abs/2406.06397) demonstrates that SCEPTR can be used for few-shot TCR specificity prediction with improved accuracy over previous methods.

SCEPTR is a BERT-like transformer-based neural network implemented in [Pytorch](https://pytorch.org).
With the default model providing best-in-class performance with only 153,108 parameters (typical protein language models have tens or hundreds of millions), SCEPTR runs fast- even on a CPU!
And if your computer does have a [CUDA-enabled GPU](https://en.wikipedia.org/wiki/CUDA), the sceptr package will automatically detect and use it, giving you blazingly fast performance without the hassle.

sceptr's API exposes three intuitive functions: `calc_vector_representations`, `calc_cdist_matrix`, and `calc_pdist_vector`- and it's all you need to make full use of the SCEPTR models.
What's even better is that they are fully compliant with [pyrepseq](https://pyrepseq.readthedocs.io)'s [tcr_metric](https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.metric.tcr_metric.TcrMetric) API, so sceptr will fit snugly into the rest of your repertoire analysis workflow.

## Installation

```bash
pip install sceptr
```
