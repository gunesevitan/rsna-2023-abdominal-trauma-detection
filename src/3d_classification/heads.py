import torch.nn as nn


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
