
# coding:utf8
from torch import nn
from .basic_module import BasicModule


class AlexNet(BasicModule):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.model_name = 'alexnet'
        self.model = models.AlexNet(pretainer = Ture)
        self.model.num_classes = num_classes 
	
	num_fts = self.model.classifier[6].in_features
	self.model.classifier[6] = nn.Linear(num_fts, num_classes)
 
    def forward(self, x):
	return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        return Adam(self.model.parameters(), lr, weight_decay)
