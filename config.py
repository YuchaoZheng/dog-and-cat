import warnings
import torch as t

class DefaultConfig(object):

    env = 'deault' # visdom环境
    vis_port = 8097
    model = 'AlexNet'  #是用模型，名字必须与models/__int__.py的名字一致

    train_data_root = './data/train' #训练集存放路径
    test_data_root = './data/test1' #测试集存放路径
    load_model_path = None #家在预训练的模型的路径，为Node代表不加在
    save_model_epoch = 1
    
    batch_size = 32
    use_gpu = True
    num_workers = 4
    print_freq = 20 #print info every N batch

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001 # initial learning rate
    lr_decay = 0.5 # 下降幅度
    weight_decay = 1e-5

    #更新函数，根据字典更新配置参数。
    def parse(self, kwargs):
        '''
        根据字典kwargs更新config参数
        '''

        #更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s"%k)
            setattr(self, k, v)
    
        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

opt = DefaultConfig()
