
###OFA-RCAN model
Data=../data
Model=../float/
PretrainModel=../float/pretrained/model_37.06505_0.04326.pth.tar

S=2 #scale
R=1 #the number of groups
F=16 #output channel
SAVE=OFA_RCAN_Finetune

CUDA_VISIBLE_DEVICES=0 python train.py --dir_data ${Data} --model RCAN_wo_CA --lr 5e-4 --save ${SAVE} --scale ${S} --n_resgroups ${R} --n_feats ${F} --n_GPUs 1 --pretrained ${PretrainModel}

