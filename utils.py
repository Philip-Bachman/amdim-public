import math
import torch
import torch.nn as nn
import torch.nn.init as init

from mixed_precision import maybe_half


def test_model(model, test_loader, device, stats, max_evals=200000):
    '''
    Evaluate accuracy on test set
    '''
    # warm up batchnorm stats based on current model
    _warmup_batchnorm(model, test_loader, device, batches=50, train_loader=False)

    def get_correct_count(lgt_vals, lab_vals):
        # count how many predictions match the target labels
        max_lgt = torch.max(lgt_vals.cpu().data, 1)[1]
        num_correct = (max_lgt == lab_vals).sum().item()
        return num_correct

    # evaluate model on test_loader
    model.eval()
    correct_glb_mlp = 0.
    correct_glb_lin = 0.
    total = 0.
    for _, (images, labels) in enumerate(test_loader):
        if total > max_evals:
            break
        images = images.to(device)
        labels = labels.cpu()
        with torch.no_grad():
            res_dict = model(x1=images, x2=images, class_only=True)
            lgt_glb_mlp, lgt_glb_lin = res_dict['class']
        # check classification accuracy
        correct_glb_mlp += get_correct_count(lgt_glb_mlp, labels)
        correct_glb_lin += get_correct_count(lgt_glb_lin, labels)
        total += labels.size(0)
    acc_glb_mlp = correct_glb_mlp / total
    acc_glb_lin = correct_glb_lin / total
    model.train()
    # record stats in the provided stat tracker
    stats.update('test_acc_glb_mlp', acc_glb_mlp, n=1)
    stats.update('test_acc_glb_lin', acc_glb_lin, n=1)


def _warmup_batchnorm(model, data_loader, device, batches=100, train_loader=False):
    '''
    Run some batches through all parts of the model to warmup the running
    stats for batchnorm layers.
    '''
    model.train()
    for i, (images, _) in enumerate(data_loader):
        if i == batches:
            break
        if train_loader:
            images = images[0]
        images = images.to(device)
        _ = model(x1=images, x2=images, class_only=True)


def flatten(x):
    return x.reshape(x.size(0), -1)


def random_locs_2d(x, k_hot=1):
    '''
    Sample a k-hot mask over spatial locations for each set of conv features
    in x, where x.shape is like (n_batch, n_feat, n_x, n_y).
    '''
    # assume x is (n_batch, n_feat, n_x, n_y)
    x_size = x.size()
    n_batch = x_size[0]
    n_locs = x_size[2] * x_size[3]
    idx_topk = torch.topk(torch.rand((n_batch, n_locs)), k=k_hot, dim=1)[1]
    khot_mask = torch.zeros((n_batch, n_locs)).scatter_(1, idx_topk, 1.)
    rand_locs = khot_mask.reshape((n_batch, 1, x_size[2], x_size[3]))
    rand_locs = maybe_half(rand_locs)
    return rand_locs


def init_pytorch_defaults(m, version='041'):
    '''
    Apply default inits from pytorch version 0.4.1 or 1.0.0.

    pytorch 1.0 default inits are wonky :-(
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

