import os
import torch

from model import Model


class Checkpointer():
    def __init__(self, output_dir=None):
        # set output dir will this checkpoint will save itself
        self.output_dir = output_dir
        self.classifier_epoch = 0
        self.classifier_step = 0
        self.info_epoch = 0
        self.info_step = 0

    def track_new_model(self, model):
        self.model = model
      
    def restore_model_from_checkpoint(self, cpt_path, training_classifier=False):
        ckp = torch.load(cpt_path)
        hp = ckp['hyperparams']
        params = ckp['model']
        self.info_epoch = ckp['cursor']['info_epoch']
        self.info_step = ckp['cursor']['info_step']
        self.classifier_epoch = ckp['cursor']['classifier_epoch']
        self.classifier_step = ckp['cursor']['classifier_step']
        self.model = Model(ndf=hp['ndf'], n_classes=hp['n_classes'], n_rkhs=hp['n_rkhs'],
                           n_depth=hp['n_depth'], encoder_size=hp['encoder_size'])
        skip_classifier = (training_classifier and self.classifier_step == 0)
        if training_classifier and self.classifier_step == 0:
            # If we are beginning the classifier training phase, we want to start
            # with a clean classifier
            model_dict = self.model.state_dict()
            partial_params = {k: v for k, v in params.items() if not k.startswith("evaluator.")}
            model_dict.update(partial_params)
            params = model_dict
        self.model.load_state_dict(params)


        print("***** CHECKPOINTING *****\n"
                "Model restored from checkpoint.\n"
                "Self-supervised training epoch {}\n"
                "Classifier training epoch {}\n"
                "*************************"
                .format(self.info_epoch, self.classifier_epoch))
        return self.model

    def _get_state(self):
        return {
            'model': self.model.state_dict(),
            'hyperparams': self.model.hyperparams,
            'cursor': {
                'info_epoch': self.info_epoch,
                'info_step': self.info_step,
                'classifier_epoch': self.classifier_epoch,
                'classifier_step':self.classifier_step,
            }
        }

    def _save_cpt(self):
        f_name = 'amdim_cpt.pth'
        cpt_path = os.path.join(self.output_dir, f_name)
        # write updated checkpoint to the desired path
        torch.save(self._get_state(), cpt_path)
        return


    def update(self, epoch, step, classifier=False):
        if classifier:
            self.classifier_epoch = epoch
            self.classifier_step = step
        else:
            self.info_epoch = epoch
            self.info_step = step
        self._save_cpt()


    def get_current_position(self, classifier=False):
        if classifier:
            return self.classifier_epoch, self.classifier_step
        return self.info_epoch, self.info_step
