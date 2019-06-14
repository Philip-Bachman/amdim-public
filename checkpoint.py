import os

import torch


class Checkpoint():

    def __init__(self, model, checkpoint_path, output_dir, fine_tuning):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.fine_tuning = fine_tuning
        # set initial state
        self.model = model
        self.info_epochs = 0
        self.info_steps = 0
        self.class_epochs = 0
        self.class_steps = 0

        # load checkpoint from disk (if available)
        self._load_checkpoint()

    def _get_state(self):
        return {
            'info_epochs': self.info_epochs,
            'info_steps': self.info_steps,
            'class_epochs': self.class_epochs,
            'class_steps': self.class_steps,
            'model': self.model.state_dict()
        }

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self._restore_model(checkpoint)
            self.info_epochs = checkpoint['info_epochs']
            self.info_steps = checkpoint['info_steps']
            self.class_epochs = checkpoint['class_epochs']
            self.class_steps = checkpoint['class_steps']
            print("***** CHECKPOINTING *****\n"
                  "Model restored from checkpoint.\n"
                  "Self-supervised training epoch {}\n"
                  "Fine-tuning epoch {}\n"
                  "*************************"
                  .format(self.info_epochs, self.class_epochs))
        else:
            print("***** CHECKPOINTING *****\n"
                  "No checkpoint found. Starting training with fresh weights.\n"
                  "*************************")

    def _get_output_path(self, fine_tuning):
        if fine_tuning:
            f_name = 'amdim_finetuned_cpt.pth'
        else:
            f_name = 'amdim_cpt.pth'
        return os.path.join(self.output_dir, f_name)

    def _restore_model(self, ckp):
        """Restore a model from the parameters of a checkpoint

        Arguments:
            ckp {OrderedDict} -- Checkpoint dict
        """
        params = ckp['model']
        if self.fine_tuning and ckp['class_steps'] == 0:
            # When starting classifier training, we want a fresh evaluator,
            # so we do not restore it from the checkpoint.
            model_dict = self.model.state_dict()
            partial_params = {k: v for k, v in params.items() if not k.startswith("evaluator.")}
            model_dict.update(partial_params)
            params = model_dict
        # load params into model
        self.model.load_state_dict(params)

    def update(self, epoch, step, fine_tuning):
        if not fine_tuning:
            self.info_epochs = epoch
            self.info_steps = step
        else:
            self.class_epochs = epoch
            self.class_steps = step
        # write updated checkpoint to the desired path
        torch.save(self._get_state(), self._get_output_path(fine_tuning))

    def get_current_position(self, fine_tuning):
        if not fine_tuning:
            return self.info_epochs, self.info_steps
        return self.class_epochs, self.class_steps
