from pytorch_nndct.apis import torch_quantizer
import torch
import os
from tqdm import tqdm
import utility
import data
import model
import loss
from option import args
import metric
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

def prepare(a, b, device):
    def _prepare(tensor):
        return tensor.to(device)

    return _prepare(a), _prepare(b)

def fast_finetune_model(model, loader, loss, device):
    model.eval()
    loss.start_log()
    with torch.no_grad():
        for batch, (lr, hr, _,) in enumerate(loader.loader_train):
            lr, hr = prepare(lr, hr, device)
            sr = model(lr, 0)
            Loss = loss(sr, hr)
    loss.end_log(len(loader.loader_train))

def calib_model_train(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch, (lr, hr, _,) in enumerate(loader.loader_train):
            lr, hr = prepare(lr, hr, device)
            sr = model(lr, 0)


def calib_model_test(model, loader, device):
    model.eval()
    self_scale = [2]
    with torch.no_grad():
        for idx_data, d in enumerate(loader.loader_test):
            for idx_scale, scale in enumerate(self_scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = prepare(lr, hr, device)
                    sr = model(lr, idx_scale)
                    sr = utility.quantize(sr, 255)

def test_model(model, loader, device):
    torch.set_grad_enabled(False)
    model.eval()
    self_scale = [2]
    timer_test = utility.timer()
    for idx_data, d in enumerate(loader.loader_test):
        eval_ssim = 0
        eval_psnr = 0
        for idx_scale, scale in enumerate(self_scale):
            d.dataset.set_scale(idx_scale)
            for lr, hr, filename in tqdm(d, ncols=80):
                lr, hr = prepare(lr, hr, device)
                sr = model(lr, idx_scale)
                sr = utility.quantize(sr, 255)
                eval_psnr += metric.calc_psnr(
                    sr, hr, scale, 255, benchmark=d)
                eval_ssim += metric.calc_ssim(
                    sr, hr, scale, 255, dataset=d)

            mean_ssim = eval_ssim / len(d)
            mean_psnr = eval_psnr /  len(d)


        print("psnr: %s, ssim: %s"%(mean_psnr, mean_ssim))
    return mean_psnr, mean_ssim

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        file_path = os.path.join(args.float_model_path, 'model_float.pt')
        pretrain_state_dict = torch.load(file_path)
        new_dict = {}
        for key in pretrain_state_dict.keys():
            new_key = 'model.' + key
            new_dict[new_key] = pretrain_state_dict[key]
        model_state_dict = model.state_dict()
        model_state_dict.update(new_dict)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        input = torch.randn([64, 3, 48, 48])
        if args.quant_mode == 'float':
            quant_model = model
            test_model(quant_model, loader, device)
        elif args.quant_mode == 'calib':
            from pytorch_nndct.apis import torch_quantizer
            quantizer = torch_quantizer(args.quant_mode, model, (input), device=device)
            quant_model = quantizer.quant_model
            if args.fast_finetune:
                _loss = loss.Loss(args, checkpoint)
                _loss = _loss.to(device)
                quantizer.fast_finetune(fast_finetune_model, (quant_model, loader, _loss, device))
            calib_model_train(quant_model, loader, device)
            quantizer.export_quant_config()
        else:
            from pytorch_nndct.apis import torch_quantizer
            if args.dump_xmodel:
                device = torch.device("cpu")
                model = model.to(device)
                input = torch.randn([1, 3, 360, 640])
                quantizer = torch_quantizer(args.quant_mode, model, (input), device=device)
                quant_model = quantizer.quant_model
                if args.fast_finetune:
                    quantizer.load_ft_param()
                output = quant_model(input, 2)
                quantizer.export_xmodel(output_dir='quantize_result/',deploy_check=True)
            else:
                quantizer = torch_quantizer(args.quant_mode, model, (input), device=device)
                quant_model = quantizer.quant_model
                if args.fast_finetune:
                    quantizer.load_ft_param()
                test_model(quant_model, loader, device)

if __name__ == '__main__':
    main()
