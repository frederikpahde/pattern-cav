import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from model_training.training_utils import get_optimizer, get_loss
from utils.metrics import get_accuracy, get_f1, get_auc
from torchvision.models import ResNet, EfficientNet, VGG, VisionTransformer
from timm.models.resnet import ResNet as ResNetTimm
from timm.models.rexnet import RexNet
from timm.models.vision_transformer import VisionTransformer as VisionTransformerTimm
from timm.models.swin_transformer import SwinTransformer
from timm.models.metaformer import MetaFormer

class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr, ckpt_path):
        self.optim = get_optimizer(optim_name, params, lr, ckpt_path)

    def set_loss(self, loss_name, weights=None):
        self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self):
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[80, 120], gamma=0.1)
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [self.optim], [scheduler]

    @staticmethod
    def get_accuracy(y_hat, y):
        return get_accuracy(y_hat, y)

    @staticmethod
    def get_f1(y_hat, y):
        return get_f1(y_hat, y)

    @staticmethod
    def get_auc(y_hat, y):
        return get_auc(y_hat, y)

    def state_dict(self, **kwargs):
        return self.model.state_dict()


class Vanilla(LitClassifier):
    def __init__(self, model, config):
        super().__init__(model, config)

    def configure_callbacks(self):
        return [
            Freeze(stop_at_layer=self.config['layer_name'])
        ]

def get_lnames_sorted_resnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    while "layer" not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(4):
        while f"layer{lidx+1}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == 3 else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == 3 else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_rexnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    while "features." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1
    lnames_sorted.append("stem_identity")
    for lidx in range(17):
        while f"features.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == 16 else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == 16 else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_efficientnet(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = ["input_identity"]
    idx = 0
    num_blocks = max([int(n.split(".")[1]) for n in lnames if "features." in n])

    while not "features.0" in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(num_blocks + 1):
        while f"features.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = "last_conv" if lidx == num_blocks else f"identity_{lidx}"
        name_relu = "last_relu" if lidx == num_blocks else f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if "identity" not in lnames[idx] and "last_" not in lnames[idx] and "stem" not in lnames[idx]:
            lnames_sorted.append(lnames[idx])
        idx += 1
        
    return lnames_sorted

def get_lnames_sorted_vgg(model):
    return ["input_identity"] + [n for n, _ in model.named_modules()][:-1]

def get_lnames_sorted_vit(model):
    return [n for n, _ in model.named_modules()][:-3] + ["correction_layer"] + [n for n, _ in model.named_modules()][-3:-1]

def get_lnames_sorted_vit_timm(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = []
    idx = 0
    while "blocks." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(12):
        while f"blocks.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = f"identity_{lidx}"
        name_relu = f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if lnames[idx] == "head":
            lnames_sorted.append("inspection_layer")
        if not any([l in lnames[idx] for l in ["identity", "inspection"]]):
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_swinformer(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = []
    idx = 0
    while "layers." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(4):
        while f"layers.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = f"identity_{lidx}"
        name_relu = f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if lnames[idx] == "head":
            lnames_sorted.append("inspection_layer")
        if not any([l in lnames[idx] for l in ["identity", "inspection"]]):
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted_metaformer(model):
    lnames = [n for n, _ in model.named_modules()]
    lnames_sorted = []
    idx = 0
    while "stages." not in lnames[idx]:
        lnames_sorted.append(lnames[idx])
        idx += 1

    for lidx in range(4):
        while f"stages.{lidx}" in lnames[idx]:
            lnames_sorted.append(lnames[idx])
            idx += 1
        name_identity = f"identity_{lidx}"
        name_relu = f"relu_{lidx}"
        lnames_sorted += [name_identity, name_relu]

    while idx < len(lnames):
        if lnames[idx] == "head":
            lnames_sorted.append("inspection_layer")
        if not any([l in lnames[idx] for l in ["identity", "inspection"]]):
            lnames_sorted.append(lnames[idx])
        idx += 1
    return lnames_sorted

def get_lnames_sorted(model):
    if isinstance(model, ResNet):
        return get_lnames_sorted_resnet(model)
    elif isinstance(model, ResNetTimm):
        return get_lnames_sorted_resnet(model)
    elif isinstance(model, RexNet):
        return get_lnames_sorted_rexnet(model)
    elif isinstance(model, EfficientNet):
        return get_lnames_sorted_efficientnet(model)
    elif isinstance(model, VGG):
        return get_lnames_sorted_vgg(model)
    elif isinstance(model, VisionTransformer):
        return get_lnames_sorted_vit(model)
    elif isinstance(model, VisionTransformerTimm):
        return get_lnames_sorted_vit_timm(model)
    elif isinstance(model, SwinTransformer):
        return get_lnames_sorted_swinformer(model)
    elif isinstance(model, MetaFormer):
        return get_lnames_sorted_metaformer(model)
    else:
        raise NotImplementedError(f"not implemented for model {model.__class__}")

class Freeze(Callback):
    def __init__(self, stop_at_layer=None, stop_before=False):
        super().__init__()
        self.stop_at_layer = stop_at_layer
        self.stop_before = stop_before

        self.freeze_types = [
            torch.nn.BatchNorm2d,
            torch.nn.Conv2d,
            torch.nn.Linear,
            torch.nn.LayerNorm,
            torch.nn.MultiheadAttention
        ]

    def check_freeze_layer(self, layer_type):
        ## FREEZE ALL LAYER TYPES FOR VIT
        # return True
        for freeze_type in self.freeze_types:
            if isinstance(layer_type, freeze_type):
                return True
        return False

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"Freezing conv+bn layers. Up to {self.stop_at_layer}")
        lnames_sorted = get_lnames_sorted(pl_module.model)
        for n, m in pl_module.model.named_modules():
            freeze_layer = lnames_sorted.index(self.stop_at_layer) >= lnames_sorted.index(n)
            if freeze_layer and self.check_freeze_layer(m):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                print(f"Freeze {n}")

        ## Freeze extra ViT layers
        params_blacklist = [
            # ViT
            "class_token", "encoder.pos_embedding",
            "cls_token", "pos_embed",

            # SwinFormer
            "relative_position_bias_table",

            # MetaFormer
            ".scale", "act.bias", "act1.bias"
            ]
        for n, p in pl_module.model.named_parameters():
            if any([p in n for p in params_blacklist]):
                p.requires_grad = False

        layers_to_optimize = [n for n, m in pl_module.model.named_parameters() if m.requires_grad]
        print(f"Done. Optimizing {layers_to_optimize}")