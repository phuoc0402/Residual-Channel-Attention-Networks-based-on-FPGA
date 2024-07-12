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

import os
import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
#for qat
from pytorch_nndct import nn as nndct_nn
from pytorch_nndct import QatProcessor
import tqdm
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def load_float_weight(model, weight):
    if os.path.isfile(weight):
        checkpoint_state_dict = torch.load(weight)
        values = []
        for key, value in checkpoint_state_dict.items():
            values.append(value)
        
        model_state_dict = model.state_dict()
        
        new_dict = {}
        i = 0
        for key, value in model_state_dict.items():
            new_dict[key] = values[i]
            i = i + 1
        model_state_dict.update(new_dict)        
        model.load_state_dict(model_state_dict)
    else:
        print("=> no checkpoint found at '{}'".format(weight))
    return model  

def main():
    global model
    global best_psnr
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if args.qat_step == 1:
                     
            #### Step1: QAT (including load float model and PQ information)
            load_float_weight(model, '../float/model_float.pt')
            dummy_input = torch.randn([args.batch_size, 3, 48, 48], dtype=torch.float32).to(device)
            qat_processor = QatProcessor(model, dummy_input, bitwidth=8)
            quantized_model = qat_processor.trainable_model()
 
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None

            t = Trainer(args, loader, quantized_model, _loss, checkpoint)
            best_psnr = t.test()
            while not t.terminate():
                t.train()
                psnr = t.test()
                if psnr > best_psnr:  
                    best_psnr = psnr
                    torch.save(quantized_model.state_dict(), os.path.join('../snapshot/', args.save, 'model/best_quantized_model.pth'))
            torch.save(quantized_model.state_dict(), os.path.join('../snapshot/', args.save, 'model/last_quantized_model.pth'))
            checkpoint.done()
            
        elif args.qat_step == 2: 
            #######Step2: Get deployable model.
            output_dir = os.path.join('../snapshot/', args.save, 'model', 'qat_result')

            dummy_input = torch.randn([args.batch_size, 3, 48, 48], dtype=torch.float32).to(device)
            qat_processor = QatProcessor(model, dummy_input, bitwidth=8)
            quantized_model = qat_processor.trainable_model()

            quantized_model.load_state_dict(torch.load(os.path.join('../snapshot/', args.save, 'model/best_quantized_model.pth')))
            deployable_net = qat_processor.convert_to_deployable(quantized_model, output_dir=output_dir)
            torch.save(deployable_net.state_dict(), os.path.join('../snapshot/', args.save, 'model/deploy_latest.pth')) 
            
        else:
            ######Step3: Export Xir model
            from pytorch_nndct.apis import torch_quantizer
            from nndct_shared.utils import NndctOption, option_util, NndctDebugLogger, NndctScreenLogger
            option_util.set_option_value("nndct_param_corr", False)
            option_util.set_option_value("nndct_equalization", False)

            output_dir = os.path.join('../snapshot/', args.save, 'model', 'qat_result')
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            model.load_state_dict(torch.load(os.path.join('../snapshot/', args.save, 'model/deploy_latest.pth')))
            input = torch.randn([1, 3, 48, 48], dtype=torch.float32).to(device)
            quantizer = torch_quantizer('test', model, (input), output_dir=os.path.join(output_dir, 'test'), device=device)  
            quant_model = quantizer.quant_model
            t = Trainer(args, loader, quant_model, _loss, checkpoint)
            t.test()
            
            ### dump xmodel
            device = torch.device("cpu")
            dump_input = torch.randn([1, 3, 360, 640], dtype=torch.float32).to(device)
            quantizer = torch_quantizer('test', model, (dump_input), output_dir=os.path.join(output_dir, 'test'), device=device)  
            output = quantizer.quant_model(dump_input, 2)
            quantizer.export_xmodel(output_dir, deploy_check=True)
                   
if __name__ == '__main__':
    main()
