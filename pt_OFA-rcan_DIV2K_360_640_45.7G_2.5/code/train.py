import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def get_model_flops(model_name, model, input_W, input_H):
    from torchprofiler import Profiler

    # Initialize your profiler
    profiler = Profiler(model)

    # Run for specified input shape
    profiler.run((1,3,input_H, input_W)) # Include batch_size. e.g. (1, 3, 224, 224)

    # Print summary
    profiler.print_summary()

    # You can also view the overall statistics respectively
    profiler.total_input
    profiler.total_output
    profiler.total_params
    profiler.total_flops
    profiler.trainable_params

    print('Model name:', model_name)

def main():
    global model

    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)

        checkpoint_state_dict = torch.load(args.pretrained)

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
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, _loss, checkpoint)
        while not t.terminate():
            t.test()
            t.train()
        checkpoint.done()

if __name__ == '__main__':
    main()
