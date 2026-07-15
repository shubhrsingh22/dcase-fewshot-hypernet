"""ProtoNet encoder and graph-hypernetwork (H-Proto) for few-shot detection.

Reproduces thesis Chapter 3.3:

* ``ProtoNet`` (Sec 3.3.3): four conv blocks (64 filters, 3x3, BN, ReLU, 2x2
  max-pool), an adaptive pool that fixes the spatial size, and a linear layer to
  a 256-d embedding.
* ``GraphHyperNet`` (Sec 3.3.4, Eqs 3.26-3.28): a graph-convolutional
  hypernetwork that builds a k-NN graph over the H*W feature-map locations,
  runs two GCN layers (C -> C/2 -> C/4), mean-pools the node embeddings, and
  produces sigmoid channel-gating coefficients that modulate the feature map.
  Inserted after conv block(s) given by ``hyper_placement`` (1-indexed):
  ``[]`` -> plain ProtoNet baseline, ``[3]`` -> H-Proto-3, ``[1,2,3,4]`` -> All.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        # ceil_mode keeps the time dim >=1 for the short, variable-length
        # segments used at validation time (avoids collapsing to size 0).
        nn.MaxPool2d(2, ceil_mode=True),
    )


class GraphHyperNet(nn.Module):
    """Graph-based hypernetwork producing channel-wise gating for a feature map."""

    def __init__(self, channels: int, knn: int = 5, max_nodes: int = 256):
        super().__init__()
        self.knn = knn
        # Cap the number of graph nodes (H*W) for tractable, memory-safe M x M
        # graph ops; the validation segments can be long (M ~ 1000s of nodes).
        self.max_nodes = max_nodes
        c1, c2 = channels // 2, channels // 4
        self.gcn1 = nn.Linear(channels, c1)
        self.gcn2 = nn.Linear(c1, c2)
        self.gate = nn.Linear(c2, channels)

    @staticmethod
    def _normalised_adj(nodes, knn):
        """Symmetric-normalised adjacency of a k-NN graph over nodes.

        nodes: (B, M, C) -> returns  D^-1/2 (A+I) D^-1/2  as (B, M, M), using
        broadcast scaling (no dense diagonal matrices).
        """
        b, m, _ = nodes.shape
        k = min(knn, m - 1) if m > 1 else 0
        dist = torch.cdist(nodes, nodes)  # (B, M, M)
        A = torch.zeros(b, m, m, device=nodes.device, dtype=nodes.dtype)
        if k > 0:
            idx = dist.topk(k + 1, dim=-1, largest=False).indices[..., 1:]
            A.scatter_(-1, idx, 1.0)
            A = torch.maximum(A, A.transpose(1, 2))  # symmetric
        eye = torch.eye(m, device=nodes.device, dtype=nodes.dtype).unsqueeze(0)
        A_hat = A + eye
        d_inv_sqrt = A_hat.sum(-1).clamp(min=1e-12).pow(-0.5)  # (B, M)
        return d_inv_sqrt.unsqueeze(2) * A_hat * d_inv_sqrt.unsqueeze(1)

    def forward(self, feat_map):
        # feat_map: (B, C, H, W)
        b, c, h, w = feat_map.shape
        fm = feat_map
        if h * w > self.max_nodes:
            s = int(self.max_nodes ** 0.5)
            fm = F.adaptive_avg_pool2d(feat_map, (min(h, s), min(w, s)))
        nodes = fm.flatten(2).transpose(1, 2)            # (B, M, C)
        norm = self._normalised_adj(nodes, self.knn)     # (B, M, M)
        z = F.relu(norm @ self.gcn1(nodes))              # (B, M, C/2)
        z = F.relu(norm @ self.gcn2(z))                  # (B, M, C/4)
        v = z.mean(dim=1)                                # (B, C/4)
        g = torch.sigmoid(self.gate(v))                  # (B, C)
        return feat_map * g.view(b, c, 1, 1)


class ProtoNet(nn.Module):
    """Thesis ProtoNet embedding network with optional graph-hypernetwork gating."""

    def __init__(self, n_filters=64, emb_dim=256, pool_out=(1, 4),
                 hyper_placement=(), hyper_knn=5, hyper_max_nodes=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            conv_block(1, n_filters),
            conv_block(n_filters, n_filters),
            conv_block(n_filters, n_filters),
            conv_block(n_filters, n_filters),
        ])
        self.hyper_placement = set(int(p) for p in hyper_placement)
        self.hypernets = nn.ModuleDict({
            str(p): GraphHyperNet(n_filters, knn=hyper_knn, max_nodes=hyper_max_nodes)
            for p in self.hyper_placement
        })
        self.pool = nn.AdaptiveAvgPool2d(tuple(pool_out))
        self.fc = nn.Linear(n_filters * pool_out[0] * pool_out[1], emb_dim)

    def forward(self, x):
        # x: (num_samples, seq_len, mel_bins)
        num, seq_len, mel = x.shape
        x = x.view(-1, 1, seq_len, mel)
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in self.hyper_placement:
                x = self.hypernets[str(i)](x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    """Residual block used by the DCASE baseline ResNet-9 encoder (Model.py)."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride, ceil_mode=True)
        self.downsample = downsample
        self.drop_rate = drop_rate

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNet(nn.Module):
    """DCASE baseline ResNet encoder (from deep_learning/Model.py); emb dim 512."""

    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.inplanes = 1
        self.layer1 = self._make_layer(64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(128, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(64, stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(64, stride=2, drop_rate=drop_rate)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        layer = _BasicBlock(self.inplanes, planes, stride, downsample, drop_rate)
        self.inplanes = planes
        return layer

    def forward(self, x):
        num, seq_len, mel = x.shape
        x = x.view(-1, 1, seq_len, mel)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # layer4 is unused in the baseline forward (kept for parity).
        x = self.pool(x)
        return x.view(x.size(0), -1)


def build_encoder(conf):
    if conf.model.encoder == "resnet":
        return ResNet()
    return ProtoNet(
        n_filters=conf.model.n_filters,
        emb_dim=conf.model.emb_dim,
        pool_out=tuple(conf.model.pool_out),
        hyper_placement=list(conf.model.hyper_placement),
        hyper_knn=conf.model.hyper_knn,
        hyper_max_nodes=conf.model.get("hyper_max_nodes", 256),
    )
