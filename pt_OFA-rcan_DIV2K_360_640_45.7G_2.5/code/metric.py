import math
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import skimage.measure
import skimage.color


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)





def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        print("the dimention of sr image is not equal to hr's! ")
        sr = sr[:,:,:hr.size(-2),:hr.size(-1)]
    diff = (sr - hr).data.div(rgb_range)

    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


import numpy as np
from scipy import signal
#from skimage.measure import compare_ssim

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
  """
  2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
  Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
  """
  m,n = [(ss-1.)/2. for ss in shape]
  y,x = np.ogrid[-m:m+1,-n:n+1]
  h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
  h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
  sumh = h.sum()
  if sumh != 0:
    h /= sumh
  return h

def calc_ssim(X, Y, scale, rgb_range, dataset=None, sigma=1.5, K1=0.01, K2=0.03, R=255):
  '''
  X : y channel (i.e., luminance) of transformed YCbCr space of X
  Y : y channel (i.e., luminance) of transformed YCbCr space of Y
  Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
  Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
  The authors of EDSR use MATLAB's ssim as the evaluation tool, 
  thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
  '''
  gaussian_filter = matlab_style_gauss2D((11, 11), sigma)

  if True:#dataset and dataset.dataset.benchmark:
    shave = scale
    if X.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        X = X.mul(convert).sum(dim=1)
        Y = Y.mul(convert).sum(dim=1)
  else:
    shave = scale + 6

  X = X[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64) 
  Y = Y[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)

  window = gaussian_filter

  ux = signal.convolve2d(X, window, mode='same', boundary='symm')
  uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

  uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
  uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
  uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')

  vx = uxx - ux * ux
  vy = uyy - uy * uy
  vxy = uxy - ux * uy

  C1 = (K1 * R) ** 2
  C2 = (K2 * R) ** 2

  A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
  D = B1 * B2
  S = (A1 * A2) / D
  mssim = S.mean()

  return mssim

def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)


def make_dual_optimizer(opt, dual_models):
    dual_optimizers = []
    for dual_model in dual_models:
        temp_dual_optim = torch.optim.Adam(
            params=dual_model.parameters(),
            lr = opt.lr, 
            betas = (opt.beta1, opt.beta2),
            eps = opt.epsilon,
            weight_decay=opt.weight_decay)
        dual_optimizers.append(temp_dual_optim)
    
    return dual_optimizers


def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.epochs),
        eta_min=opt.eta_min
    )

    return scheduler


def make_dual_scheduler(opt, dual_optimizers):
    dual_scheduler = []
    for i in range(len(dual_optimizers)):
        scheduler = lrs.CosineAnnealingLR(
            dual_optimizers[i],
            float(opt.epochs),
            eta_min=opt.eta_min
        )
        dual_scheduler.append(scheduler)

    return dual_scheduler


def init_model(args):
    # Set the templates here
    if args.model.find('DRN-S') >= 0:
        if args.scale == 4:
            args.n_blocks = 30
            args.n_feats = 16
        elif args.scale == 8:
            args.n_blocks = 30
            args.n_feats = 8
        else:
            print('Use defaults n_blocks and n_feats.')
        args.dual = True

    if args.model.find('DRN-L') >= 0:
        if args.scale == 4:
            args.n_blocks = 40
            args.n_feats = 20
        elif args.scale == 8:
            args.n_blocks = 36
            args.n_feats = 10
        else:
            print('Use defaults n_blocks and n_feats.')
        args.dual = True


