import torch.nn as nn
import monai.networks.nets

from heads import ClassificationHead


class MONAIConvolutionalClassificationModel(nn.Module):

    def __init__(self, backbone_class, backbone_args, dropout_rate, freeze_parameters):

        super(MONAIConvolutionalClassificationModel, self).__init__()

        self.backbone = getattr(monai.networks.nets, backbone_class)(**backbone_args)

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        input_features = self.backbone._fc.in_features
        self.backbone._dropout = nn.Dropout(dropout_rate)
        self.backbone._fc = nn.Identity()
        self.backbone._swish = nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features)

    def forward(self, x):

        x = self.backbone(x)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output


class MONAITransformerClassificationModel(nn.Module):

    def __init__(self, backbone_class, backbone_args, freeze_parameters):

        super(MONAITransformerClassificationModel, self).__init__()
        self.backbone = getattr(monai.networks.nets, backbone_class)(**backbone_args)

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        input_features = self.backbone.classification_head[0].in_features
        self.backbone.classification_head = nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features)

    def forward(self, x):

        x = self.backbone(x)[0]
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output
