import os
import torch


class Checkpoint():
    def __init__(self, model, cpt_load_path, output_dir):
        # set output dir will this checkpoint will save itself
        self.output_dir = output_dir
        # set initial state
        self.model = model
        self.info_epochs = 0
        self.info_steps = 0
        # load checkpoint from disk (if available)
        self._load_cpt(cpt_load_path)

    def _get_state(self):
        return {
            'info_epochs': self.info_epochs,
            'info_steps': self.info_steps,
            'class_epochs': self.class_epochs,
            'class_steps': self.class_steps,
            'model': self.model.state_dict()
        }

    def _load_cpt(self):
        if os.path.isfile(self.cpt_load_path):
            checkpoint = torch.load(self.cpt_load_path)
            self._restore_model(checkpoint)
            self.info_epochs = checkpoint['info_epochs']
            self.info_steps = checkpoint['info_steps']
            print("***** CHECKPOINTING *******************\n"
                  "Model restored from checkpoint.\n"
                  "- stopped at encoder training epoch {}\n"
                  "***************************************"
                  .format(self.info_epochs))
        else:
            print("***** CHECKPOINTING ****************\n"
                  "No checkpoint found. Starting fresh.\n"
                  "************************************")

    def _save_cpt(self):
        f_name = 'amdim_cpt.pth'
        cpt_path = os.path.join(self.output_dir, f_name)
        # write updated checkpoint to the desired path
        torch.save(self._get_state(), cpt_path)
        return

    def _restore_model(self, checkpoint):
        """Restore a model from the parameters of a checkpoint

        Arguments:
            checkpoint {OrderedDict} -- Checkpoint dict
        """
        params = checkpoint['model']
        # load params into model
        self.model.load_state_dict(params)

    def update(self, epoch, step):
        self.info_epochs = epoch
        self.info_steps = step
        self._save_cpt()

    def get_current_position(self):
        return self.info_epochs, self.info_steps
