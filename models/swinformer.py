import torch
import timm

from utils.lrp_canonizers import VITCanonizer

def get_swin_base_22k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_swinformer("timm/swin_base_patch4_window7_224.ms_in22k", ckpt_path, pretrained, n_class)

def get_swin_base_22k_ft1k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_swinformer("timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k", ckpt_path, pretrained, n_class)

def get_swinformer(model_name, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)

    if n_class and n_class != 1000:
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, n_class)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.inspection_layer = torch.nn.Identity()
    for i in range(len(model.layers)):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())

    model.forward_features = _forward_features.__get__(model)
    model.forward = _forward.__get__(model)

    return model

def _forward_features(self, x):
        x = self.patch_embed(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if hasattr(self, f"identity_{i}"):
                x = getattr(self, f"identity_{i}")(x)
        # x = self.layers(x)
        x = self.norm(x)
        return x

def _forward(self, x):
    x = self.forward_features(x)
    x = self.inspection_layer(x)
    x = self.forward_head(x)
    return x

def get_swinformer_canonizer():
    return [VITCanonizer()]
