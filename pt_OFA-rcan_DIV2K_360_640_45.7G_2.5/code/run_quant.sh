
###RCAN-S model
export CUDA_VISIBLE_DEVICES=1
Model=../float/
Data=../data/

S=2 #scale
R=1 #the number of groups
F=16 #output channel

# Note: for this model, direct 8bit-quantization gets an accuracy that is not so satisfactory, so we use the fast-finetune trick
echo "Calibrating model quantization..."
MODE='calib'

python test.py --dir_data ${Data} --model RCAN_wo_CA --scale ${S} --n_resgroups ${R} --n_feats ${F} --data_test Set5+Set14+B100+Urban100  --float_model_path ${Model} --quant_mode ${MODE} --fast_finetune --nndct_finetune_lr_factor 0.015


echo "Testing quantized model..."
MODE='test'

python test.py --dir_data ${Data} --model RCAN_wo_CA --scale ${S} --n_resgroups ${R} --n_feats ${F} --data_test Set5+Set14+B100+Urban100  --float_model_path ${Model} --quant_mode ${MODE} --fast_finetune


#echo "dump xmodel..."
#MODE='test'

#python test.py --dir_data ${Data} --model RCAN_wo_CA --scale ${S} --n_resgroups ${R} --n_feats ${F} --data_test Set5+Set14+B100+Urban100  --float_model_path ${Model} --quant_mode ${MODE} --fast_finetune --dump_xmodel
