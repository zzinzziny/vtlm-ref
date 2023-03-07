# VTLM
VTLM (Visaul Translation Language Modeling) is Basically, the translation language modeling (TLM) structure that concatenates vision features to form embeddings.
It extends the translation language modeling with masked region classification, and pre-train with three-way parallel vision & language corpora.
Then, lastly fine-tuning for multimodal machine translation.

### Checkpoints
[here!](https://zenodo.org/record/4646961/files/vtlm_eacl21_checkpoints.tar.bz2)

![Screen Shot 2023-02-02 at 13.40.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/13681f18-db5f-4729-a0aa-5159a1a74407/Screen_Shot_2023-02-02_at_13.40.44.png)

### Requirements
- Python3
- Numpy
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)
- `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch`
- `git clone https://github.com/ImperialNLP/VTLM.git`
- `pip install -r requirements.txt`

### Datasets
- Multi30k
- Conceptual Captions (CC)

## How to use
1. Preparing the data (code path: `data/`)
  a. download
  ```
  $ cd data
  bash scripts/00-download.sh
  ```
2. Pre-processing the corpora (code path: `data/`)
tokenize + BPE-ize
