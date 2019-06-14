import os
import argparse

import torch

import mixed_precision
from stats import StatTracker
from datasets import build_dataset, Dataset, get_dataset
from model import Model
from checkpoint import Checkpoint
from task_self_supervised import train_self_supervised
from task_fine_tune import train_fine_tune

parser = argparse.ArgumentParser(description='Infomax Representations -- Self-Supervised Training')
parser.add_argument('--dataset', type=str, default='STL10')
parser.add_argument('--batch_size', type=int, default=200,
                    help='input batch size (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')

# parameters for model and training objective
parser.add_argument('--finetune', action='store_true', default=False,
                    help="Wether to run self-supervised or"
                    "finetuning training task")
parser.add_argument('--ndf', type=int, default=128,
                    help='feature width for network')
parser.add_argument('--n_rkhs', type=int, default=1024,
                    help='number of dimensions in fake RKHS embeddings')
parser.add_argument('--tclip', type=float, default=20.0,
                    help='soft clipping value for NCE scores')
parser.add_argument('--res_depth', type=int, default=3)


# parameters for output, logging, checkpointing, etc
parser.add_argument('--output_dir', type=str, default='./',
                    help='directory where tensorboard events and checkpoints will be stored')
parser.add_argument('--input_dir', type=str, default='/mnt/imagenet',
                    help="Input directory for the dataset. Not needed For C10,"
                    " C100 or STL10 as the data will be automatically downloaded.")
parser.add_argument('--checkpoint_path', type=str, default='amdim_cpt.pth',
                    help='Path to the checkpoint to restart from.')

args = parser.parse_args()


def main():
    # create target output dir if it doesn't exist yet
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = get_dataset(args.dataset)

    # get a helper object for tensorboard logging
    stat_tracker = StatTracker(log_dir=args.output_dir)

    # get dataloaders for training and testing
    train_loader, test_loader, num_classes = \
        build_dataset(dataset=dataset,
                      batch_size=args.batch_size,
                      input_dir=args.input_dir,
                      fine_tuning=args.finetune)

    torch_device = torch.device('cuda')
    # create new model with random parameters
    model = Model(ndf=args.ndf, n_classes=num_classes, n_rkhs=args.n_rkhs,
                  tclip=args.tclip, res_depth=args.res_depth, dataset=dataset)
    # restore model parameters from a checkpoint if requested
    checkpoint = \
        Checkpoint(model, args.checkpoint_path, args.output_dir, args.finetune)
    model = model.to(torch_device)

    if args.finetune:
        # run the classifier training/finetuning task
        task = train_fine_tune
    else:
        # run the self-supervised task for training encoder
        task = train_self_supervised
    # do the real stuff...
    task(model, args.learning_rate, dataset, train_loader,
         test_loader, stat_tracker, checkpoint, args.output_dir, torch_device)


if __name__ == "__main__":
    print(args)
    main()
