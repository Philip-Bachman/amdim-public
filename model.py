import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixed_precision import maybe_half
from utils import flatten, random_locs_2d, Flatten
from costs import LossMultiNCE


def has_many_gpus():
    return torch.cuda.device_count() >= 6


class Encoder(nn.Module):
    def __init__(self, dummy_batch, num_channels=3, ndf=64, n_rkhs=512, 
                n_depth=3, encoder_size=32, use_bn=False):
        super(Encoder, self).__init__()
        self.ndf = ndf
        self.n_rkhs = n_rkhs
        self.use_bn = use_bn
        self.dim2layer = None

        # encoding block for local features
        print('Using a {}x{} encoder'.format(encoder_size, encoder_size))
        if encoder_size == 32:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResNxN(ndf, ndf, 1, 1, 0, use_bn),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 4, True, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 4, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 4, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 64:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        elif encoder_size == 128:
            self.layer_list = nn.ModuleList([
                Conv3x3(num_channels, ndf, 5, 2, 2, False, pad_mode='reflect'),
                Conv3x3(ndf, ndf, 3, 1, 0, False),
                ConvResBlock(ndf * 1, ndf * 2, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 2, ndf * 4, 4, 2, 0, n_depth, use_bn),
                ConvResBlock(ndf * 4, ndf * 8, 2, 2, 0, n_depth, use_bn),
                MaybeBatchNorm2d(ndf * 8, True, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResBlock(ndf * 8, ndf * 8, 3, 1, 0, n_depth, use_bn),
                ConvResNxN(ndf * 8, n_rkhs, 3, 1, 0, use_bn),
                MaybeBatchNorm2d(n_rkhs, True, True)
            ])
        else:
            raise RuntimeError("Could not build encoder."
                               "Encoder size {} is not supported".format(encoder_size))
        self._config_modules(dummy_batch, [1, 5, 7], n_rkhs, use_bn)

    def init_weights(self, init_scale=1.):
        '''
        Run custom weight init for modules...
        '''
        for layer in self.layer_list:
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
        for layer in self.modules():
            if isinstance(layer, (ConvResNxN, ConvResBlock)):
                layer.init_weights(init_scale)
            if isinstance(layer, FakeRKHSConvNet):
                layer.init_weights(init_scale)

    def _config_modules(self, x, rkhs_layers, n_rkhs, use_bn):
        '''
        Configure the modules for extracting fake rkhs embeddings for infomax.
        '''
        enc_acts = self._forward_acts(x)
        self.dim2layer = {}
        for i, h_i in enumerate(enc_acts):
            for d in rkhs_layers:
                if h_i.size(2) == d:
                    self.dim2layer[d] = i
        # get activations and feature sizes at different layers
        self.ndf_1 = enc_acts[self.dim2layer[1]].size(1)
        self.ndf_5 = enc_acts[self.dim2layer[5]].size(1)
        self.ndf_7 = enc_acts[self.dim2layer[7]].size(1)
        # configure modules for fake rkhs embeddings
        self.rkhs_block_1 = NopNet()
        self.rkhs_block_5 = FakeRKHSConvNet(self.ndf_5, n_rkhs, use_bn)
        self.rkhs_block_7 = FakeRKHSConvNet(self.ndf_7, n_rkhs, use_bn)

    def _forward_acts(self, x):
        '''
        Return activations from all layers.
        '''
        # run forward pass through all layers
        layer_acts = [x]
        for _, layer in enumerate(self.layer_list):
            layer_in = layer_acts[-1]
            layer_out = layer(layer_in)
            layer_acts.append(layer_out)
        # remove input from the returned list of activations
        return_acts = layer_acts[1:]
        return return_acts

    def forward(self, x):
        '''
        Compute activations and Fake RKHS embeddings for the batch.
        '''
        if has_many_gpus():
            if x.abs().mean() < 1e-4:
                r1 = torch.zeros((1, self.n_rkhs, 1, 1),
                                 device=x.device, dtype=x.dtype).detach()
                r5 = torch.zeros((1, self.n_rkhs, 5, 5),
                                 device=x.device, dtype=x.dtype).detach()
                r7 = torch.zeros((1, self.n_rkhs, 7, 7),
                                 device=x.device, dtype=x.dtype).detach()
                return r1, r5, r7
        # compute activations in all layers for x
        acts = self._forward_acts(x)
        # gather rkhs embeddings from certain layers
        r1 = self.rkhs_block_1(acts[self.dim2layer[1]])
        r5 = self.rkhs_block_5(acts[self.dim2layer[5]])
        r7 = self.rkhs_block_7(acts[self.dim2layer[7]])
        return r1, r5, r7


class Evaluator(nn.Module):
    def __init__(self, n_classes, ftr_1=None, dim_1=None):
        super(Evaluator, self).__init__()
        if ftr_1 is None:
            # rely on provided input feature dimensions
            self.dim_1 = dim_1
        else:
            # infer input feature dimensions from provided features
            self.dim_1 = ftr_1.size(1)
        self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(self.dim_1, self.n_classes, n_hidden=1024, p=0.2)
        self.block_glb_lin = \
            MLPClassifier(self.dim_1, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        h_top_cls = flatten(ftr_1).detach()
        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp, lgt_glb_lin


class Model(nn.Module):
    def __init__(self, ndf, n_classes, n_rkhs, tclip=20.,
                 n_depth=3, encoder_size=32, use_bn=False):
        super(Model, self).__init__()
        self.hyperparams = {
            'ndf': ndf,
            'n_classes': n_classes,
            'n_rkhs': n_rkhs,
            'tclip': tclip, 
            'n_depth': n_depth,
            'encoder_size': encoder_size,
            'use_bn': use_bn
        }

        # self.n_rkhs = n_rkhs
        self.tasks = ('1t5', '1t7', '5t5', '5t7', '7t7')
        dummy_batch = torch.zeros((2, 3, encoder_size, encoder_size))

        # encoder that provides multiscale features
        self.encoder = Encoder(dummy_batch, num_channels=3, ndf=ndf, 
                               n_rkhs=n_rkhs, n_depth=n_depth,
                               encoder_size=encoder_size, use_bn=use_bn)
        rkhs_1, rkhs_5, _ = self.encoder(dummy_batch)
        # convert for multi-gpu use
        self.encoder = nn.DataParallel(self.encoder)

        # configure hacky multi-gpu module for infomax costs
        self.g2l_loss = LossMultiNCE(tclip=tclip)

        # configure modules for classification with self-supervised features
        self.evaluator = Evaluator(n_classes, ftr_1=rkhs_1)

        # gather lists of self-supervised and classifier modules
        self.info_modules = [self.encoder.module, self.g2l_loss]
        self.class_modules = [self.evaluator]

    def init_weights(self, init_scale=1.):
        self.encoder.module.init_weights(init_scale)

    def encode(self, x, no_grad=True, use_eval=False):
        '''
        Encode the images in x, with or without grads detached.
        '''
        if use_eval:
            self.eval()
        x = maybe_half(x)
        if no_grad:
            with torch.no_grad():
                rkhs_1, rkhs_5, rkhs_7 = self.encoder(x)
        else:
            rkhs_1, rkhs_5, rkhs_7 = self.encoder(x)
        if use_eval:
            self.train()
        return maybe_half(rkhs_1), maybe_half(rkhs_5), maybe_half(rkhs_7)

    def reset_evaluator(self, n_classes=None):
        '''
        Reset the evaluator module, e.g. to apply encoder on new data.
        - evaluator is reset to have n_classes classes (if given)
        '''
        dim_1 = self.evaluator.dim_1
        if n_classes is None:
            n_classes = self.evaluator.n_classes
        self.evaluator = Evaluator(n_classes, dim_1=dim_1)
        self.class_modules = [self.evaluator]
        return self.evaluator

    def forward(self, x1, x2, class_only=False):
        '''
        Input:
          x1 : images from which to extract features -- x1 ~ A(x)
          x2 : images from which to extract features -- x2 ~ A(x)
          class_only : whether we want all outputs for infomax training
        Output:
          res_dict : various outputs depending on the task
        '''
        # dict for returning various values
        res_dict = {}
        if class_only:
            # shortcut to encode one image and evaluate classifier
            rkhs_1, _, _ = self.encode(x1, no_grad=True)
            lgt_glb_mlp, lgt_glb_lin = self.evaluator(rkhs_1)
            res_dict['class'] = [lgt_glb_mlp, lgt_glb_lin]
            res_dict['rkhs_glb'] = flatten(rkhs_1)
            return res_dict

        # hack for redistributing workload in highly multi-gpu setting
        # -- yeah, "highly-multi-gpu" is obviously subjective...
        if has_many_gpus():
            n_batch = x1.size(0)
            n_gpus = torch.cuda.device_count()
            assert (n_batch % (n_gpus - 1) == 0), 'n_batch: {}'.format(n_batch)
            # expand input with dummy chunks so cuda:0 can skip compute
            chunk_size = n_batch // (n_gpus - 1)
            dummy_chunk = torch.zeros_like(x1[:chunk_size])
            x1 = torch.cat([dummy_chunk, x1], dim=0)
            x2 = torch.cat([dummy_chunk, x2], dim=0)

        # run augmented image pairs through the encoder
        r1_x1, r5_x1, r7_x1 = self.encoder(x1)
        r1_x2, r5_x2, r7_x2 = self.encoder(x2)

        # hack for redistributing workload in highly-multi-gpu setting
        if has_many_gpus():
            # strip off dummy vals returned by cuda:0
            r1_x1, r5_x1, r7_x1 = r1_x1[1:], r5_x1[1:], r7_x1[1:]
            r1_x2, r5_x2, r7_x2 = r1_x2[1:], r5_x2[1:], r7_x2[1:]

        # compute NCE infomax objective at multiple scales
        loss_1t5, loss_1t7, loss_5t5, lgt_reg = \
            self.g2l_loss(r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2)
        res_dict['g2l_1t5'] = loss_1t5
        res_dict['g2l_1t7'] = loss_1t7
        res_dict['g2l_5t5'] = loss_5t5
        res_dict['lgt_reg'] = lgt_reg
        # grab global features for use elsewhere
        res_dict['rkhs_glb'] = flatten(r1_x1)

        # compute classifier logits for online eval during infomax training
        # - we do this for both images in each augmented pair...
        lgt_glb_mlp, lgt_glb_lin = self.evaluator(ftr_1=torch.cat([r1_x1, r1_x2]))
        res_dict['class'] = [lgt_glb_mlp, lgt_glb_lin]
        return res_dict


##############################
# Layers for use in model... #
##############################


class MaybeBatchNorm2d(nn.Module):
    def __init__(self, n_ftr, affine, use_bn):
        super(MaybeBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class NopNet(nn.Module):
    def __init__(self, norm_dim=None):
        super(NopNet, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x):
        if self.norm_dim is not None:
            x_norms = torch.sum(x**2., dim=self.norm_dim, keepdim=True)
            x_norms = torch.sqrt(x_norms + 1e-6)
            x = x / x_norms
        return x


class Conv3x3(nn.Module):
    def __init__(self, n_in, n_out, n_kern, n_stride, n_pad,
                 use_bn=True, pad_mode='constant'):
        super(Conv3x3, self).__init__()
        assert(pad_mode in ['constant', 'reflect'])
        self.n_pad = (n_pad, n_pad, n_pad, n_pad)
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(n_in, n_out, n_kern, n_stride, 0,
                              bias=(not use_bn))
        self.relu = nn.ReLU(inplace=True)
        self.bn = MaybeBatchNorm2d(n_out, True, use_bn) if use_bn else None

    def forward(self, x):
        if self.n_pad[0] > 0:
            # pad the input if required
            x = F.pad(x, self.n_pad, mode=self.pad_mode)
        # conv is always applied
        x = self.conv(x)
        # apply batchnorm if required
        if self.bn is not None:
            x = self.bn(x)
        # relu is always applied
        out = self.relu(x)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits

class FakeRKHSConvNet(nn.Module):
    def __init__(self, n_input, n_output, use_bn=False):
        super(FakeRKHSConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
                               padding=0, bias=False)
        # BN is optional for hidden layer and always for output layer
        self.bn_hid = MaybeBatchNorm2d(n_output, True, use_bn)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        # initialize shortcut to be like identity (if possible)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)
        return

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)
        return

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn_hid(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h


class ConvResNxN(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, use_bn=False):
        super(ConvResNxN, self).__init__()
        assert (n_out >= n_in)
        self.n_in = n_in
        self.n_out = n_out
        self.width = width
        self.stride = stride
        self.pad = pad
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_in, n_out, width, stride, pad, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 1, 1, 0, bias=False)
        self.conv3 = None
        # ...
        self.bn1 = MaybeBatchNorm2d(n_out, True, use_bn)
        return

    def init_weights(self, init_scale=1.):
        # initialize first conv in res branch
        # -- rescale the default init for nn.Conv2d layers
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        # initialize second conv in res branch
        # -- set to 0, like fixup/zero init
        nn.init.constant_(self.conv2.weight, 0.)
        return

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        h2 = self.conv2(self.relu2(h1))
        if (self.n_out < self.n_in):
            h3 = self.conv3(x)
        elif (self.n_in == self.n_out):
            h3 = F.avg_pool2d(x, self.width, self.stride, self.pad)
        else:
            h3_pool = F.avg_pool2d(x, self.width, self.stride, self.pad)
            h3 = F.pad(h3_pool, (0, 0, 0, 0, 0, self.n_out - self.n_in))
        h23 = h2 + h3
        return h23


class ConvResBlock(nn.Module):
    def __init__(self, n_in, n_out, width, stride, pad, depth, use_bn):
        super(ConvResBlock, self).__init__()
        layer_list = [ConvResNxN(n_in, n_out, width, stride, pad, use_bn)]
        for i in range(depth - 1):
            layer_list.append(ConvResNxN(n_out, n_out, 1, 1, 0, use_bn))
        self.layer_list = nn.Sequential(*layer_list)
        return

    def init_weights(self, init_scale=1.):
        '''
        Do a fixup-style init for each ConvResNxN in this block.
        '''
        for m in self.layer_list:
            m.init_weights(init_scale)
        return

    def forward(self, x):
        # run forward pass through the list of ConvResNxN layers
        x_out = self.layer_list(x)
        return x_out

