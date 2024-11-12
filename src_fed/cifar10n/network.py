import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common.typing import NDArrays
from torch import nn, einsum
from torch.optim.optimizer import Optimizer, required
import torch.utils.checkpoint as checkpoint
import itertools
import collections
import warnings
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
       # embedding_output = out.view(out.size(0), -1)
       # print(embedding_output.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        embedding_output = out
        out = self.linear(out)
        return out, embedding_output

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def get_network(num_classes=10, weights_fp=None):
    model = ResNet18(num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=5, patch_size=2, num_classes=10):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.gelu = nn.GELU()
        self.initial_bn = nn.BatchNorm2d(dim)

        # Create the sequence of Mixer layers
        self.mixer_layers = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.gelu(out)
        out = self.initial_bn(out)

        out = self.mixer_layers(out)

        # Retrieve embedding before passing through the final linear layer
        out = self.flatten(self.avg_pool(out))
        embedding_output = out
        logits = self.linear(out)

        return logits, embedding_output

def get_convmixer_network(num_classes=10, weights_fp=None):
    model = ConvMixer(dim=256, depth=8, num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model


def swish(x):
    return x * x.sigmoid()

def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        embedding_output = out
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out, embedding_output


def EfficientNetB0(num_classes):
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg=cfg, num_classes=num_classes)

def get_efficientnetb0_network(num_classes=10, weights_fp=None):
    model = EfficientNetB0(num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model


class VGG(nn.Module):
    def __init__(self, layer_spec, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()

        layers = []
        in_channels = 3
        for l in layer_spec:
            if l == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU(),
                ]
                in_channels = l

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, None

def vgg16(num_classes=10, init_weights=False):
    vgg16_cfg = [
        64,
        64,
        "pool",
        128,
        128,
        "pool",
        256,
        256,
        256,
        "pool",
        512,
        512,
        512,
        "pool",
        512,
        512,
        512,
        "pool",
    ]
    return VGG(vgg16_cfg, num_classes, init_weights)


""" ViT-small  """

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), None


def get_vitsmall_network(num_classes=10, weights_fp=None):
    model = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model


def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), depth = ind + 1),
                LayerScale(dim, PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0]), None


def get_cait_network(num_classes=10, weights_fp=None):
    model = CaiT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

class ResNet20(nn.Module):
    def __init__(self, input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
                 shortcut_type="B", block_type="preactivated", first_conv={"filters": 16, "kernel_size": 3, "stride": 1},
                 dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True, embeddings_dim=None):
        super(ResNet20, self).__init__()
        self.l2_reg = l2_reg
        self.shortcut_type = shortcut_type
        self.cardinality = cardinality
        self.dropout = dropout
        self.preact_shortcuts = preact_shortcuts
        self.bootleneck_width = bootleneck_width
        self.embeddings_dim = embeddings_dim
        
        self.first_conv = Conv2dSamePadding(in_channels=input_shape[0], out_channels=first_conv["filters"], 
                                            kernel_size=first_conv["kernel_size"], stride=first_conv["stride"], bias=False)
        
        self.blocks = nn.ModuleList()
        
        block_types = {
            "preactivated": self.preactivation_block,
            "bootleneck": self.bootleneck_block,
            "original": self.original_block
        }
        self.selected_block = block_types[block_type]

        in_channels = first_conv["filters"]
        for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
            for i in range(group_size):
                stride = stride if i == 0 else 1
                self.blocks.append(self.selected_block(in_channels, feature, stride, block_idx, i))
                in_channels = feature
        
        self.bn_relu = self.bn_relu_fn()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(features[-1], n_classes)

        if embeddings_dim:
            self.embedding = nn.Linear(features[-1], embeddings_dim)

    def bn_relu_fn(self):
        return nn.Sequential(
            nn.BatchNorm2d(num_features=self.first_conv.out_channels),
            nn.ReLU()
        )

    def shortcut(self, x, out_channels, stride, mode):
        in_channels = x.shape[1]
        if in_channels == out_channels:
            return x
        elif mode == "B":
            return self.regularized_padded_conv(out_channels, kernel_size=1, stride=stride)(x)
        elif mode == "B_original":
            x = self.regularized_padded_conv(out_channels, kernel_size=1, stride=stride)(x)
            return nn.BatchNorm2d(out_channels)(x)
        elif mode == "A":
            if stride > 1:
                x = F.avg_pool2d(x, kernel_size=1, stride=stride)
            pad = (0, 0, 0, out_channels - in_channels)
            return F.pad(x, pad, "constant", 0)
        else:
            raise KeyError("Parameter shortcut_type not recognized!")

    def regularized_padded_conv(self, out_channels, kernel_size, stride=1):
        return Conv2dSamePadding(
            in_channels=self.first_conv.out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def original_block(self, in_channels, out_channels, stride, block_idx, sub_block_idx):
        def block_fn(x):
            x = self.regularized_padded_conv(out_channels, kernel_size=3, stride=stride)(x)
            x = self.bn_relu_fn()(x)
            x = self.regularized_padded_conv(out_channels, kernel_size=3)(x)
            x = nn.BatchNorm2d(out_channels)(x)

            mode = "B_original" if self.shortcut_type == "B" else self.shortcut_type
            shortcut = self.shortcut(x, out_channels, stride, mode)
            x = x + shortcut
            return nn.ReLU()(x)
        return block_fn

    def preactivation_block(self, in_channels, out_channels, stride, block_idx, sub_block_idx):
        def block_fn(x):
            x = self.bn_relu_fn()(x)
            if sub_block_idx == 0:
                x = self.first_conv(x)
            x = self.regularized_padded_conv(out_channels, kernel_size=3, stride=stride)(x)
            if self.dropout > 0:
                x = nn.Dropout(self.dropout)(x)
            x = self.bn_relu_fn()(x)
            x = self.regularized_padded_conv(out_channels, kernel_size=3)(x)
            shortcut = self.shortcut(x, out_channels, stride, self.shortcut_type)
            x = x + shortcut
            return x
        return block_fn

    def bootleneck_block(self, in_channels, out_channels, stride, block_idx, sub_block_idx):
        def block_fn(x):
            x = self.bn_relu_fn()(x)
            if sub_block_idx == 0:
                x = self.first_conv(x)
            x = self.regularized_padded_conv(out_channels // self.bootleneck_width, kernel_size=1)(x)
            x = self.bn_relu_fn()(x)
            x = self.regularized_padded_conv(out_channels // self.bootleneck_width, kernel_size=3, stride=stride)(x)
            x = self.bn_relu_fn()(x)
            x = self.regularized_padded_conv(out_channels, kernel_size=1)(x)
            shortcut = self.shortcut(x, out_channels, stride, self.shortcut_type)
            x = x + shortcut
            return x
        return block_fn

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.embeddings_dim:
            embedding = self.embedding(x)
            return x, embedding
        return x, None


def get_resnet20_network(num_classes=10, weights_fp=None):
    input_shape = (32, 32, 3)
    model = load_model(input_shape=input_shape, num_classes=num_classes)
    if weights_fp is not None:
        model.load_state_dict(torch.load(weights_fp))
    return model
