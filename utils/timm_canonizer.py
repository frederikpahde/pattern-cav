from timm.models._efficientnet_blocks import (
    SqueezeExcite, 
    DepthwiseSeparableConv, 
    InvertedResidual
)
from timm.layers import BatchNormAct2d
from zennit.canonizers import AttributeCanonizer, CompositeCanonizer
import torch
from zennit import layer as zlayer
from zennit.canonizers import SequentialMergeBatchNorm

class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2):
        return x1*x2 

    @staticmethod
    def backward(ctx,grad_output):
        return torch.zeros_like(grad_output), grad_output
    


class BatchNormActCanonizer(AttributeCanonizer):
    '''Canonizer specifically for SE layers.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, BatchNormAct2d):
            new_bn = torch.nn.BatchNorm2d(module.num_features).eval()
            new_bn.bias.data = module.bias.data
            new_bn.weight.data = module.weight.data
            new_bn.running_mean = module.running_mean
            new_bn.running_var = module.running_var
            attributes = {
                'forward': cls.forward.__get__(module),
                'bn': new_bn
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified SE forward for SENetworks.'''
        x = self.bn(x)
        x = self.drop(x)
        x = self.act(x)
        return x

class SqueezeExciteModuleCanonizer(AttributeCanonizer):
    '''Canonizer specifically for SE layers.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcite):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified SE forward for SENetworks.'''
        identity = x
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # out = identity * self.gate(x_se)
        out = self.fn_gate.apply(self.gate(x_se), identity)
        return out
    
class DepthwiseSeparableConvCanonizer(AttributeCanonizer):
    '''Canonizer specifically for DepthwiseSeparableConv of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, DepthwiseSeparableConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None
    
    @staticmethod
    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_skip:
            x = self.drop_path(x)
            x = torch.stack([shortcut, x], dim=-1)
            x = self.canonizer_sum(x)

        return x

class InvertedResidualCanonizer(AttributeCanonizer):
    '''Canonizer specifically for InvertedResidual of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, InvertedResidual):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)

        # if self.has_residual:
        if self.has_skip:
            # if self.drop_path_rate > 0.:
            x = self.drop_path(x)
            x = torch.stack([shortcut, x], dim=-1)
            x = self.canonizer_sum(x)
        return x


class EfficientNetCanonizer(CompositeCanonizer):
    def __init__(self):
        super().__init__((

            SqueezeExciteModuleCanonizer(),
            InvertedResidualCanonizer(),
            DepthwiseSeparableConvCanonizer(),
            BatchNormActCanonizer(),
            SequentialMergeBatchNorm(),
        ))
