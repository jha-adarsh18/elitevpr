"""The event-native student: the only thing deployed."""

import timm
import torch.nn as nn

from .pooling import POOLINGS, build_pooler


class EventViTStudent(nn.Module):
    """Event histogram -> one L2-normalisable global descriptor.

    `teacher_dim` is the width of the student's own per-patch projection, not
    the teacher's width -- a legacy name kept because every checkpoint and both
    evaluators read it. Widening it to the teacher's 8448 would make the
    per-patch projection (B, 576, 8448), 622 MB a forward.
    """

    def __init__(self, backbone_name="vit_small_patch16_dinov3.lvd1689m",
                 teacher_dim=1024, num_patches=576, img_size=384, in_channels=3,
                 pooling="clamp", pretrained=True):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

        self.num_patches = num_patches
        self.pooling = pooling

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, img_size=img_size)
        # Learned input normalisation: the student is fed raw histogram values,
        # not ImageNet-normalised ones.
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(self.backbone.embed_dim, teacher_dim)
        if pooling == "clamp":
            self.gem = build_pooler("clamp")
        else:
            self.pool = build_pooler(pooling)

    @property
    def pooler(self):
        return self.gem if self.pooling == "clamp" else self.pool

    def forward(self, x):
        x = self.input_norm(x)
        features = self.backbone.forward_features(x)
        patches = features[:, -self.num_patches:, :]
        return self.pooler(self.proj(patches))
