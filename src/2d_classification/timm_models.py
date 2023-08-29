import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ClassificationHead(nn.Module):

    def __init__(self, input_dimensions, pooling_type):

        super(ClassificationHead, self).__init__()

        self.pooling_type = pooling_type
        self.bowel_head = nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, 1, bias=True)
        self.extravasation_head = nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, 1, bias=True)
        self.kidney_head = nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, 3, bias=True)
        self.liver_head = nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, 3, bias=True)
        self.spleen_head = nn.Linear(input_dimensions * 2 if pooling_type == 'concat' else input_dimensions, 3, bias=True)

    def forward(self, x):

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        bowel_output = self.bowel_head(x)
        extravasation_output = self.extravasation_head(x)
        kidney_output = self.kidney_head(x)
        liver_output = self.liver_head(x)
        spleen_output = self.spleen_head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output


class TimmConvolutionalClassificationModel(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, freeze_parameters, head_args):

        super(TimmConvolutionalClassificationModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        input_features = self.backbone.get_classifier().in_features
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)
        bowel_output, extravasation_output, kidney_output, liver_output, spleen_output = self.head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output
