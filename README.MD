
# XPPG-PCA

This code can be used to evaluate the speech severity of pathological speakers.

Please decide first whether you need dynamic or static usage before using the code. (You will most likely need dynamic)

## Table of Contents

- [Dynamic vs Static Usage](#dynamic-vs-static-usage)
    - [Dynamic Usage](#dynamic-usage)
    - [Static Usage](#static-usage)
- [Inference](#inference)
- [Reproducing Results with Feature Extraction](#reproducing-results-with-feature-extraction)
- [Static Installation](#static-installation)
- [FAQ](#faq)
    - [Why can't I create the results using the dynamic inference code?](#why-cant-i-create-the-results-using-the-dynamic-inference-code)
    - [What are the outputs of the respective codes?](#what-are-the-outputs-of-the-respective-codes)

## Dynamic vs Static Usage

Irrespective of usage please download this first.

Static features (not only static features)
https://drive.google.com/file/d/1CKanXshGdVFB9I_OJ24hVVO_2IpcD_XA/view?usp=sharing

Models
https://drive.google.com/file/d/1Oueqi4uwpm2uMnmRJTzVasXyySB2V9_o/view?usp=sharing

There are two usages to this codebase, dynamic and static.

### Dynamic Usage

If you need to use severity evaluation for your research, or want to evaluate your own audio files, I recommend using the dynamic one. The setup of this is significantly harder.

### Static Usage

Static usage relies on pre-extracted features provided in the repository. If you want to quickly reproduce the results of the paper, follow these instructions.

## Dynamic

### Installation

To set up the environment, you will need the following dependencies:
- pykaldi
- ESPNET (only the Python library is needed for inference)
- speechbrain (for x-vector extraction)

Below is an example of how to install these dependencies:

```bash
conda create --name pykaldi_env_3 python=3.8
conda activate pykaldi_env_3
pip install numpy
# Choose the appropriate binary from the pykaldi site
wget https://github.com/pykaldi/pykaldi/releases/download/v0.2.2/pykaldi-0.2.2-cp38-cp38-linux_x86_64.whl.gz
gunzip pykaldi-0.2.2-cp38-cp38-linux_x86_64.whl.gz
pip install pykaldi-0.2.2-cp38-cp38-linux_x86_64.whl
pip install matplotlib
pip install pip==24 # workaround for the omegaconf dependency solver issues
pip install espnet
pip install speechbrain
pip install chainer==6.0.0 cupy==6.0.0
```

Note: Some parts of the Kaldi installation might require sudo access. If you do not have that, consider using containerized solutions such as Singularity or Docker.

After that make sure that the KALDI_ROOT environment variable in the path.sh
points to the correct place.

### Dynamic inferences

```bash
. path.sh
python inference.py --audio_file audio.wav
```
If you don't provide audio file, an audio file from the COPAS dataset will be fetched.

## Repeating the experiment with dynamic code

If you want to repeat (not reproduce) the experiment with the dynamic code, you will need the COPAS dataset. You can access this dataset by creating an account at the Instituut voor de Nederlandse Taalmaterialen:

[Download COPAS dataset](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

After installation, please modify the COPAS_PATH in params.py.

After modification of the COPAS_PATH, the experiment can be repeated by:

```bash
python inference.py --copas
```

## Static use case

### Static Installation

If you do not need to calculate features, a simpler environment is sufficient:

```bash
conda create --name static_xppg_pca39 python=3.9
conda activate static_xppg_pca
conda install scipy scikit-learn
pip install kaldiio
```

### Static reproduction

```bash
python static_reproduction.py
```

## FAQ

### Why can't I create the results using the dynamic inference code?

Kaldi's `compute-fbank-feats` uses dither in its default extraction. The patterns extracted are not significantly different. Even with dither turned off, there will be a 1e-6 difference in the extracted features.

### What are the outputs of the respective codes?

| Category       | Static Correlation | Dynamic Correlation |
|----------------|--------------------|---------------------|
| Voice disorder (V) | 0.9949             | 0.9949              |
| Laryngectomy (L)   | 0.8559             | 0.8506              |
| Hearing impairment (H) | 0.8098             | 0.8115              |
| Dysarthric (D)     | 0.4356             | 0.4371              |
| All                | 0.5084             | 0.5112              |
| All except D       | 0.7557             | 0.7565              |

