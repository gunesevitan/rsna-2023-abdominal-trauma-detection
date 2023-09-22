import sys
import torch.nn as nn

from heads import ClassificationHead

sys.path.append('..')
import settings

sys.path.append('../../venv/lib/python3.11/site-packages/Submission_2nd_Covid19_Competition')
from training.models.convnext import ConvNeXt3d, load_params3Dfromparams3D


class ConvNeXt3DModel(nn.Module):

    def __init__(self, backbone_args, freeze_parameters):

        super(ConvNeXt3DModel, self).__init__()

        self.backbone = ConvNeXt3d(**backbone_args)
        load_params3Dfromparams3D(
            model=self.backbone,
            pretrained_path=settings.MODELS / 'convnext_3d',
            ten_net=0,
            use_transformer=False,
            size='tiny',
            datasize=256,
            pretrained_mode='stoic'
        )

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        input_features = self.backbone.pre_head.in_features
        self.backbone.pre_head = nn.Identity()
        self.backbone.metadata_prehead = nn.Identity()
        self.backbone.head = nn.Identity()
        self.backbone.metadata_embedding = nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features)

    def forward(self, x):

        x = self.backbone.forward_features(x, age=None, sex=None)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output
