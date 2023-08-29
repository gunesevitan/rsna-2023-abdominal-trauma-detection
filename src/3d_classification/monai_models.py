import torch.nn as nn
import monai.networks.nets


class ClassificationHead(nn.Module):

    def __init__(self, input_dimensions):

        super(ClassificationHead, self).__init__()

        self.bowel_head = nn.Linear(input_dimensions, 1, bias=True)
        self.extravasation_head = nn.Linear(input_dimensions, 1, bias=True)
        self.kidney_head = nn.Linear(input_dimensions, 3, bias=True)
        self.liver_head = nn.Linear(input_dimensions, 3, bias=True)
        self.spleen_head = nn.Linear(input_dimensions, 3, bias=True)

    def forward(self, x):

        bowel_output = self.bowel_head(x)
        extravasation_output = self.extravasation_head(x)
        kidney_output = self.kidney_head(x)
        liver_output = self.liver_head(x)
        spleen_output = self.spleen_head(x)

        return bowel_output, extravasation_output, kidney_output, liver_output, spleen_output


class MONAIConvolutionalClassificationModel(nn.Module):

    def __init__(self, model_class, model_args, dropout_rate, freeze_parameters):

        super(MONAIConvolutionalClassificationModel, self).__init__()

        self.backbone = getattr(monai.networks.nets, model_class)(**model_args)

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
