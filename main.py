from torchnet import meter
from config import opt
import os
import torch
import models
from data.dataset import DogCat
from utils.visualize import Visualizer
from tqdm import tqdm

def train(**kwargs):
    '''
    训练
    '''
    opt.parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    #step1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.train()
    model.to(opt.device)
    lr = opt.lr
    #step2：数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)

    train_dataloader = torch.utils.data.DataLoader(
                   train_data, batch_size = opt.batch_size,
                   shuffle = True,
                   num_workers = opt.num_workers)

    val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size = opt.batch_size,
                shuffle = True,
                num_workers = opt.num_workers)
    
    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
            lr = opt.lr, weight_decay = opt.weight_decay)

    
    # step4：统计指标：平滑处理之后的损失，混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for index, (data, label) in tqdm(enumerate(train_dataloader)):
            #训练模型
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            #更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())

            if (index + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                        
                #debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
        
        model.save()
        
        #混淆矩阵和准确率
        val_cm, val_accuray = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuray)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                   epoch = epoch,
                   loss = loss_meter.value()[0],
                   val_cm = str(val_cm.value()),
                   train_cm = str(confusion_matrix.value()),
                   lr = lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

def val(model, input):
    '''
    验证，计算模型在验证集上的准确率等信息，用亿辅助训练
    '''
    #把模式设置为验证模式
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (data, label) in enumerate(input):
        val_input = data.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach(), label.detach())
    model.train()
   
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

'''
测试时，需要计算每个样本属于狗的概率，并将结果保存成csv文件。测试的代码与验证比较相似，但需要自己加载模型和数据。
'''
def test(**kwargs):
    '''
    测试(inference)
    '''
    opt.parse(kwargs)
    
    model = getattr(models, opt.model).eval()
    if opt.load_model_path:
        opt.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    #数据load
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = torch.utils.data.loader(train_data,
            shuffle = False,
            batch_size = opt.batch_size,
            num_workers = pot.num_workers)

    resluts = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = data.to(opt.device)
        score = model(input)
        probability = torch.nn.functional.softmax(score[:,1]).data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results.append(batch_results)
    write_csv(results, opt.result_file)
    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def help():
    '''
    打印帮助的信息: python main.py help
    '''
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} help
    avaiable args:'''.format(__file__))

    #TODO
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__ == '__main__':
    import fire
    fire.Fire()
