import visdom
import time
import numpy as np

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis =visdom.Visdom(env=env,use_incoming_socket=False, **kwargs)

        #画的第几个数，相当于横坐标
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        d: dict (name, value)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        
        #获取对应的横坐标
        x = self.index.get(name, 0)
        self.vis.line(Y = np.array([y]), X = np.array([x]),
                win = name,
                opts = dict(title = name),
                update = None if x == 0 else 'append',
                **kwargs
                )
        
        self.index[name] = x + 1
        

    def img(self, name, img, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win = 'log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time = time.strftime('%m%d_%H%M%S'),
            info = info
            ))
        self.vis.text(self.log_text, win)

    def getattr_(self, name):
        return getattr(self.vis, name)
