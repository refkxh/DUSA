#!/bin/bash
# use open3d==0.9 on centos 7
# copy tools/pypcd.py to pypcd installation path

# conda create -n opencood python=3.7
# conda activate opencood
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cython matplotlib numba numpy pyyaml scipy shapely tensorboard tensorboardx tqdm -c conda-forge
pip install cumm easydict einops open3d opencv-python pygame pypcd scikit-image timm torch-tb-profiler -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install spconv-cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
