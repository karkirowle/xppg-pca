
# XPPG-PCA

This code can be used to evaluate the speech severity of pathological speakers.

## Installation
(Many thanks for the authors of ESPNet for inspiration)

1. Please install Kaldi in advance, following the instructions in pykaldi. The installation of Kaldi can be complicated
enough so we did not include in the installation script. 
2. Please make sure the change the KALDI_ROOT in the path.sh

Next, please make after changing the location of your Conda installation in the Makefile as below

```
cd tools
make
```

If there are no errors, a testing should run with an example audio file. and the following output should appear
```
Test passed
```

## Reproduction without pre-extracted features

1. Please [download the COPAS dataset](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)
2. In the xppg_pca/params.py file, modify the COPAS_PATH to the location of COPAS dataset.
```
. path.sh
python xppg_pca/inference.py
```

## Reproduction with pre-extracted features

In case you want to get the exact results you have to do a static installation after performing the steps above.

```
conda create --name static_xppg_pca39 python=3.9
conda activate static_xppg_pca
conda install scipy scikit-learn
pip install kaldiio
```

## Running the code on your audio file

You can either use the xppg_pca/inference.py as a command line program or use xppg_pca as a python module.

An example of using it as a Python module

```
from glob import glob
from xppg_pca.inference import PPGXvectorPCAInference
from tqdm import tqdm

files = glob("dataset/**/**/*.wav", recursive=True)


ppg_pca = PPGXvectorPCAInference("models/model.last5.avg.best", 
                                     "models/ppg_xvector_pca_object_moment_1.pkl")

output_file = "nki_ccrt_sentence_clean_scores.tsv"
f = open(output_file, "w")
for file in tqdm(files):
    result = ppg_pca.predict(file)
    f.write(f"{file}\t{result}\n")

f.close()
```
## FAQ

### Why can't I create the results using the dynamic inference code?

As far as I understand, Kaldi's `compute-fbank-feats` uses dither in its default extraction. The patterns extracted are not significantly different. Even with dither turned off, there will be a 1e-6 difference in the extracted features.
In future work, we hope that we do not need to rely on the Kaldi installation to achieve good results.

### What are the outputs of the respective codes?

| Category       | Static Correlation | Dynamic Correlation |
|----------------|--------------------|---------------------|
| Voice disorder (V) | 0.9949             | 0.9949              |
| Laryngectomy (L)   | 0.8559             | 0.8506              |
| Hearing impairment (H) | 0.8098             | 0.8115              |
| Dysarthric (D)     | 0.4356             | 0.4371              |
| All                | 0.5084             | 0.5112              |
| All except D       | 0.7557             | 0.7565              |


