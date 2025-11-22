git clone https://github.com/facebookresearch/segment-anything.git
conda create -n SAM python=3.11
conda activate SAM
登录https://pytorch.org/get-started/locally/，安装pytorch和torchvision
cd segment-anything
pip install -e .
下载checkpoint 
	vit-h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
	vit-l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
	vit-b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
参照automatic_mask_generator_example.ipynb，执行Jupterbook segment-anything/notebooks/automatic_mask_generator_example.ipynb
