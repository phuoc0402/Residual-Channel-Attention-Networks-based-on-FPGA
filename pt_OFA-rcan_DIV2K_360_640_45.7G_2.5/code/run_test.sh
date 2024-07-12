
###OFA-RCAN model
Model=../float/
Data=../data/

S=2 #scale
R=1 #the number of groups
n_feats=16
MODE=float

#evaluation dense mode
CUDA_VISIBLE_DEVICES=1 python test.py --dir_data ${Data} --model RCAN_wo_CA --scale ${S} --n_resgroups ${R} --n_feats ${n_feats} --test_only --data_test Set5+Set14+B100+Urban100  --float_model_path ${Model} --quant_mode ${MODE} 

