import time
import torch

class BasicModule(torch.nn.Module):
    
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存模型：名字+时间
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.path')
        torch.save(self.state_dict(), name)
        return name
    
    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)

#TODO
class Flat(torch.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
