import torch
import timm

from utils.lrp_canonizers import VITCanonizer

def get_metaformer_s18_1k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_metaformer("caformer_s18.sail_in1k", ckpt_path, pretrained, n_class)

def get_metaformer_s18_22k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_metaformer("caformer_s18.sail_in22k", ckpt_path, pretrained, n_class)

def get_metaformer_s36_1k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_metaformer("caformer_s36.sail_in1k", ckpt_path, pretrained, n_class)

def get_metaformer_s36_22k(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_metaformer("caformer_s36.sail_in22k", ckpt_path, pretrained, n_class)

def get_metaformer(model_name, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)

    if n_class and n_class != 1000:
        model.head.fc.fc2 = torch.nn.Linear(model.head.fc.fc2.in_features, n_class)
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
    for i in range(len(model.stages) - 1):
        print("added identity", i)
        setattr(model, f"identity_{i}", torch.nn.Identity())
    model.forward_features = _forward_features.__get__(model)
    model.forward = _forward.__get__(model)

    return model

def _forward_features(self, x):
    x = self.stem(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
       raise ValueError
    else:
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if hasattr(self, f"identity_{i}"):
                x = getattr(self, f"identity_{i}")(x)
        # x = self.stages(x)
    return x

def _forward(self, x):
    x = self.forward_features(x)
    x = self.inspection_layer(x)
    x = self.forward_head(x)
    return x

def get_metaformer_canonizer():
    return [VITCanonizer()]
