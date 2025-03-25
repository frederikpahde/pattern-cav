import os

import numpy as np
import torch
from zennit.core import stabilize

from model_correction.base_correction_method import LitClassifier, Freeze
from utils.cav import compute_cav

class Clarc(LitClassifier):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)

        self.std = None
        self.layer_name = config["layer_name"]
        self.dataset_name = config["dataset_name"]
        self.cav_scope = config["cav_scope"]
        self.model_name = config["model_name"]

        assert "artifact_sample_ids" in kwargs.keys(), "artifact_sample_ids have to be passed to ClArC correction methods"
        assert "sample_ids" in kwargs.keys(), "all sample_ids have to be passed to ClArC correction methods"
        assert "class_names" in kwargs.keys(), "class_names has to be passed to ClArC correction methods"
        assert "mode" in kwargs.keys(), "mode has to be passed to ClArC correction methods"

        self.artifact_sample_ids = kwargs["artifact_sample_ids"]
        self.sample_ids = kwargs["sample_ids"]
        self.class_names = kwargs["class_names"]

        self.direction_mode = config["direction_mode"]
        self.mode = kwargs['mode']

        print(f"Using {len(self.artifact_sample_ids)} artifact samples.")

        artifact_type = config.get('artifact_type', None)
        artifact_extension = f"_{artifact_type}-{config['p_artifact']}" if artifact_type else ""
        self.path = f"{config['dir_precomputed_data']}/global_relevances_and_activations/{self.dataset_name}{artifact_extension}/{self.model_name}"

        cav, mean_length, mean_length_targets = self.compute_cav(self.mode, norm=False)
        
        self.cav = cav
        self.mean_length = mean_length
        self.mean_length_targets = mean_length_targets
        hooks = []
        for n, m in self.model.named_modules():
            if n == self.layer_name:
                print("Registered forward hook.")
                hooks.append(m.register_forward_hook(self.clarc_hook))
        self.hooks = hooks

    def compute_cav(self, mode, norm=False):
        vecs = []
        sample_ids = []

        path = self.path
        cav_scope = self.cav_scope or self.class_names

        for class_id in cav_scope:
            path_precomputed_activations = f"{path}/{self.layer_name}_class_{class_id}_all.pth"
            print(f"reading precomputed relevances/activations from {path_precomputed_activations}")
            data = torch.load(path_precomputed_activations)
            if data['samples']:
                sample_ids += data['samples']
                vecs.append(torch.stack(data[mode], 0))

        vecs = torch.cat(vecs, 0)
        sample_ids = np.array(sample_ids)

        # Only keep samples that are in self.sample_ids (usually training set)
        all_sample_ids = np.array(self.sample_ids)
        filter_sample = np.array([id in all_sample_ids for id in sample_ids])
        vecs = vecs[filter_sample]
        sample_ids = sample_ids[filter_sample]

        target_ids = np.array(
            [np.argwhere(sample_ids == id_)[0][0] for id_ in self.artifact_sample_ids if
             np.argwhere(sample_ids == id_)])
        targets = np.array([1 * (j in target_ids) for j, x in enumerate(sample_ids)])
        X = vecs.detach().cpu().numpy()
        X = X.reshape(X.shape[0], -1)
        cav = compute_cav(
            X, targets, cav_type=self.direction_mode
        )

        mean_length = (vecs[targets == 0].flatten(start_dim=1)  * cav).sum(1).mean(0)
        mean_length_targets = (vecs[targets == 1].flatten(start_dim=1) * cav).sum(1).mean(0)

        return cav, mean_length, mean_length_targets

    def clarc_hook(self, m, i, o):
        pass

    def configure_callbacks(self):
        return [Freeze(
            self.layer_name
        )]


class PClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)

        self.path = f"results/global_relevances_and_activations/{os.path.basename(config['config_file'])[:-5]}"
        if os.path.exists(self.path):
            print("Re-computing CAV.")
            cav, mean_length, mean_length_targets = self.compute_cav(self.mode)
            self.cav = cav
            self.mean_length = mean_length
            self.mean_length_targets = mean_length_targets
        else:
            if self.hooks and not kwargs.get("eval_mode", False):
                for hook in self.hooks:
                    print("Removed hook. No hook should be active for training.")
                    hook.remove()
                self.hooks = []

    def clarc_hook(self, m, i, o):
        outs = o + 0
        is_2dim = outs.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = self.cav.to(outs)
        if self.mode == "cavs_full":
            length = (outs.flatten(start_dim=1) * cav).sum(1)
        else:
            length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        beta = (cav * v).sum(1)
        mag = (self.mean_length - length).to(outs) / stabilize(beta)
        v = v.reshape(1, *outs.shape[1:]) if self.mode == "cavs_full" else v[..., None, None]
        addition = (mag[:, None, None, None] * v)
        acts = outs + addition
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts


class AClarc(Clarc):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.lamb = self.config["lamb"]  # 10

    def clarc_hook(self, m, i, o):
        outs = o + 0
        is_2dim = outs.dim() == 2
        outs = outs[..., None, None] if is_2dim else outs
        cav = self.cav.to(outs)
        if self.mode == "cavs_full":
            length = (outs.flatten(start_dim=1) * cav).sum(1)
        else:
            length = (outs.flatten(start_dim=2).max(2).values * cav).sum(1)
        v = self.cav.to(outs)
        beta = (cav * v).sum(1)
        mag = (self.mean_length_targets - length).to(outs) / stabilize(beta)
        v = v.reshape(1, *outs.shape[1:]) if self.mode == "cavs_full" else v[..., None, None]

        addition = (mag[:, None, None, None] * v).requires_grad_()
        acts = outs + addition
        acts = acts.squeeze(-1).squeeze(-1) if is_2dim else acts
        return acts
