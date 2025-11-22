git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
cp -r datasets/ checkpoints/ predict.sh ./DeepLabV3Plus-Pytorch
cd DeepLabV3Plus-Pytorch
conda create -n deeplabV3 python=3.12
conda activate deeplabV3
pip install -r requirements.txt
source predict.sh
# 分割结果在./DeepLabV3Plus-Pytorch/test_results/里