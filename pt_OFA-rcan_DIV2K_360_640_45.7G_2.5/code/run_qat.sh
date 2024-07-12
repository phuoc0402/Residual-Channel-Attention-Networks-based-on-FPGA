# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



###OFA-RCAN model
Data=../data
Model=../float/


S=2 #scale
R=1 #the number of groups
F=16 #output channel
SAVE=OFA_RCAN_QAT

CUDA_VISIBLE_DEVICES=0 python train_qat.py --qat_step 1 --dir_data ${Data} --model RCAN_wo_CA_QAT --lr 5e-3 --epochs 250 --decay 50-100-150-200-250 --data_test Set5 --save ${SAVE} --scale 2 --n_resgroups ${R} --n_feats ${F}  --batch_size 16 --n_GPUs 1 \

CUDA_VISIBLE_DEVICES=0 python train_qat.py --qat_step 2 --dir_data ${Data} --model RCAN_wo_CA_QAT --data_test Set5 --save ${SAVE} --scale 2 --n_resgroups ${R} --n_feats ${F}  --batch_size 16 --n_GPUs 1 \

CUDA_VISIBLE_DEVICES=0 python train_qat.py --qat_step 3 --dir_data ${Data} --model RCAN_wo_CA_QAT --data_test Set5+Set14+B100+Urban100 --save ${SAVE} --scale 2 --n_resgroups ${R} --n_feats ${F}  --batch_size 16 --n_GPUs 1 --test_only \

