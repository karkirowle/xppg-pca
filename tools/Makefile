ENV_NAME=xppg_test

PYTHON_VERSION=3.8
# DERIVE FROM PYTHON VERSION
PYTHON_SHORT_VERSION=$(shell echo $(PYTHON_VERSION) | sed -e 's/\.//g')
CONDA_ROOT=/data/group1/z40484r/anaconda3
# Fix the above

PYKALDI_WHL=pykaldi-0.2.2-cp$(PYTHON_SHORT_VERSION)-cp$(PYTHON_SHORT_VERSION)-linux_x86_64.whl
PYKALDI_URL=https://github.com/pykaldi/pykaldi/releases/download/v0.2.2/$(PYKALDI_WHL).gz

.PHONY: all install_pykaldi env deps clean models

all: env install_pykaldi deps models check_install

env:
	./setup_conda.sh ${CONDA_ROOT} $(ENV_NAME) $(PYTHON_VERSION)
deps:
	. ./activate_python.sh && \
	pip install numpy matplotlib pip==24 && \
	pip install espnet speechbrain && \
	pip install chainer==6.0.0 && \
	cd .. && \
	pip install -e .



install_pykaldi:
	. ./activate_python.sh && \
	pip install numpy && \
	wget --no-check-certificate $(PYKALDI_URL) && \
	gunzip $(PYKALDI_WHL).gz && \
	pip install $(PYKALDI_WHL)

models:
	rm -rf ../models
	rm -rf ../static_features
	. ./activate_python.sh && \
	pip install gdown && \
	cd .. && \
	gdown 1Oueqi4uwpm2uMnmRJTzVasXyySB2V9_o && \
	unzip models.zip && \
	gdown 1CKanXshGdVFB9I_OJ24hVVO_2IpcD_XA && \
	cd models && \
	gdown 1LRCs7YEAL-iTvKLtLch3vW0mQ8MuPHCV && \
	cd .. && unzip static_features.zip
	cd .. && rm static_features.zip
	cd .. && rm models.zip

check_install:
	. ./activate_python.sh && \
	conda activate $(ENV_NAME) && \
	cd .. && \
	python tools/check_install.py
clean:
	# Remove conda env
	. ./activate_python.sh && \
	conda deactivate && \
	conda remove --name $(ENV_NAME) --all -y
	rm -f $(PYKALDI_WHL)
	rm activate_python.sh
	rm random_noise.wav
	rm -rf ../models
	rm -rf ../static_features