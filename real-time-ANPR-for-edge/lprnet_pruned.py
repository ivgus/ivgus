# lprnet_pruned.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SmallBasicBlock, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LPRNetPrunable(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(LPRNetPrunable, self).__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            SmallBasicBlock(64, 128),
            SmallBasicBlock(128, 128),
            nn.MaxPool2d(3, 2),

            SmallBasicBlock(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SmallBasicBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), (2, 1)),

            nn.Dropout(dropout_rate),
            nn.Conv2d(256, num_classes, (1, 1), 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.backbone(x)
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(b, w, c * h)
        return features

    def apply_pruning(self, amount=0.3):
        """
        Глобальный pruning по величине весов ко всем Conv2d слоям в backbone.
        Доля обрезаемых весов (0.3 = 30%).
        """
        parameters_to_prune = []
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        print(f"Pruning applied: {amount * 100}% of weights zeroed out in Conv layers.")

    def remove_pruning_reparametrization(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        print("Pruning reparametrization removed. Model ready for export.")
