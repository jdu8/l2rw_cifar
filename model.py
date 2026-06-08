"""ResNet-32 for CIFAR-10.

Architecture follows the original He et al. CIFAR ResNet:
  n=5 blocks per stage → 6n+2 = 32 layers total.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A: zero-pad the shortcut (no extra params, matches paper)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    """ResNet-32 for CIFAR-10 (n=5 blocks per stage).

    forward(x, return_embedding=False)
        Returns logits, or (logits, embedding) when return_embedding=True.
        embedding is the 64-d GAP vector before the final linear layer.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Stage 0: 3→16, 32×32
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # Stages 1-3: n=5 blocks each
        self.layer1 = self._make_stage(16, 16, n=5, stride=1)
        self.layer2 = self._make_stage(16, 32, n=5, stride=2)
        self.layer3 = self._make_stage(32, 64, n=5, stride=2)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_planes, planes, n, stride):
        blocks = [BasicBlock(in_planes, planes, stride)]
        for _ in range(n - 1):
            blocks.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        embedding = out.view(out.size(0), -1)   # [B, 64] — penultimate repr
        logits = self.fc(embedding)
        if return_embedding:
            return logits, embedding
        return logits
