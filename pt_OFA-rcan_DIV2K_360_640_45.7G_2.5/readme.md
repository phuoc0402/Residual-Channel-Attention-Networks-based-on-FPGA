## Contents

1. [Environment](#Environment)
2. [Preparation](#Preparation)
3. [Test/Train](#Test/Train)
4. [Performance](#Performance)
5. [Model information](#Model information)
6. [Quantize](#Quantize)

## Environment
1. Environment requirement
  - Python 3.7
  - Pytorch 1.7.1

2. Installation
   - Create virtual envrionment and activate it (without docker):
   ```shell
   conda create -n torch-1.7.1 python=3.7
   conda activate torch-1.7.1
   ```
   - Activate virtual envrionment (with docker):
   ```shell
   conda vitis-ai-pytorch
   ```
   - Install all the python dependencies using pip:
   ```shell
   pip install --user -r requirements.txt
   ```
Note: If you are in the released Docker env, there is no need to create virtual envrionment.
## Preparation

1. Datasets description
   - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
   - [Benchmarks](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)
 
2. Dataset diretory structure
   + data
       + DIV2K
       + benchmark

## Test/Train

1. Model description
   ofa-rcan based on ofa search method under latency constraint (vck190 latency constraint=59ms) for scale=2x. 

2. Evaluation
   ```
   # perform evaluation on 4 benchmarks: Set5, Set14, B100, Urban100
   cd code/
   sh run_test.sh
   ```
3. Training
   ```
   # perform training on DIV2K dataset. Training is based on a pretrained model which is generated using OFA under latency constraint.
   cd code/
   sh run_train.sh
   ```
4. Quantization
   ```
   # perform qat on DIV2K dataset
   cd code/
   sh run_qat.sh
   ```

## Performance

| Method     | Scale | Set5         |  Set14        | B100 | Urban100 |
|------------|-------|--------------|---------------|----------|-------|
|OFA-RCAN (float) |X2 |37.6536 / 0.9593|33.1687 / 0.9138|31.8912 / 0.8965|30.9775 / 0.9165|
|OFA-RCAN (INT8-fastfinetune) |X2  |37.2065 / 0.9546|32.8967 / 0.9095|31.6928 / 0.8923|30.6529 / 0.9118|
|OFA-RCAN (INT8-qat) |X2  |37.3844 / 0.9560|33.0118 / 0.9105|31.7852 / 0.8935|30.8393 / 0.9126|

### Model_info

1. Data preprocess
  ```
  data channel order: BGR(0~255)                  
  input = input 
  ```
2. System Environment

The operation and accuracy provided above are verified in Ubuntu16.04.10, cuda-9.0, Driver Version: 460.32.03, GPU NVDIA P100




