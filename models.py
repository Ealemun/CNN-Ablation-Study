import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    """Somme du nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PlainBlock(nn.Module):
    """
    Bloc de base : conv 3x3 -> (BN optionnelle) -> ReLU, deux fois.
    Sert de référence "sans astuce" pour la comparaison.
    """

    def __init__(self, channels, use_bn=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class BottleneckBlock(nn.Module):
    """
    1x1 (réduit les canaux) -> 3x3 (traite, moins cher car moins de canaux)
    -> 1x1 (restaure les canaux), avec option skip connection résiduelle.
    """

    def __init__(
        self,
        channels,
        reduction=4,
        use_bn=False,
        use_skip=False
    ):
        super().__init__()

        mid = max(channels // reduction, 4)

        self.reduce = nn.Conv2d(
            channels,
            mid,
            kernel_size=1
        )

        self.process = nn.Conv2d(
            mid,
            mid,
            kernel_size=3,
            padding=1
        )

        self.expand = nn.Conv2d(
            mid,
            channels,
            kernel_size=1
        )

        self.bn_reduce = (
            nn.BatchNorm2d(mid)
            if use_bn
            else nn.Identity()
        )

        self.bn_process = (
            nn.BatchNorm2d(mid)
            if use_bn
            else nn.Identity()
        )

        self.bn_expand = (
            nn.BatchNorm2d(channels)
            if use_bn
            else nn.Identity()
        )

        self.use_skip = use_skip
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)

        out = self.process(out)
        out = self.bn_process(out)
        out = self.relu(out)

        out = self.expand(out)
        out = self.bn_expand(out)

        if self.use_skip:
            out = out + identity

        return self.relu(out)


class AblationCNN(nn.Module):
    """
    Réseau générique :
    stem -> N blocs -> Global Average Pooling -> FC.

    variant in {
        "baseline",
        "bottleneck",
        "bottleneck_bn",
        "bottleneck_bn_skip"
    }
    """

    VARIANTS = (
        "baseline",
        "bottleneck",
        "bottleneck_bn",
        "bottleneck_bn_skip"
    )

    def __init__(
        self,
        variant="baseline",
        channels=32,
        num_blocks=6,
        num_classes=10
    ):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Expected one of {self.VARIANTS}"
            )

        self.variant = variant

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        use_bn = variant in (
            "bottleneck_bn",
            "bottleneck_bn_skip"
        )

        use_skip = variant == "bottleneck_bn_skip"

        blocks = []

        for _ in range(num_blocks):
            if variant == "baseline":
                block = PlainBlock(
                    channels=channels,
                    use_bn=False
                )
            else:
                block = BottleneckBlock(
                    channels=channels,
                    reduction=4,
                    use_bn=use_bn,
                    use_skip=use_skip
                )

            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            channels,
            num_classes
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def build_model(variant, channels=32, num_blocks=6):
    """Instancie et retourne un AblationCNN."""
    return AblationCNN(
        variant=variant,
        channels=channels,
        num_blocks=num_blocks,
        num_classes=10
    )
