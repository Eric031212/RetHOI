#!/usr/bin/env bash
# Creates the `rethoi` conda environment (Python 3.10, PyTorch 2.1.2 + CUDA 11.8).
set -e
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -q '^rethoi ' || conda create -y -n rethoi python=3.10
conda activate rethoi
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" scipy==1.11.4 trimesh==4.0.10 smplx==0.1.28 matplotlib==3.7.3 \
    tqdm blobfile ftfy regex spacy==3.7.2 gdown scikit-learn einops tensorboard
# chumpy (needed to unpickle SMPL .pkl files) has a broken build and numpy>=1.24 incompatibility
pip install --no-build-isolation chumpy==0.70
CHUMPY_INIT=$(python -c "import importlib.util as u; print(u.find_spec('chumpy').submodule_search_locations[0])")/__init__.py
sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf/' "$CHUMPY_INIT"
pip install git+https://github.com/openai/CLIP.git
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
conda install -y -c conda-forge ffmpeg mesalib
# mesh rendering (pyrender + OSMesa, no display needed)
pip install h5py pyrender imageio imageio-ffmpeg "opencv-python-headless<4.11" "numpy<2"
pip install --no-deps PyOpenGL==3.1.7
python -c "import torch, chumpy, clip, smplx; print(torch.__version__, torch.cuda.is_available())"
echo ENV_OK
