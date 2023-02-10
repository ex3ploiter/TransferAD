import pandas as pd
from Attack import fgsm,attack_pgd
from util.parser import get_default_parser
from util.benchmark import fashionmnist
from util.benchmark import mnist
from util.benchmark import cifar10
from util.benchmark import *
from util.helpers.setup import checkpoint, make_dirs, newline, save_model_info, to_gpu
from util.helpers.log import Log
from sklearn.metrics import average_precision_score, roc_auc_score
from nn.resnet import resnet26
from nn.helpers.metrics import accuracy
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from fgsm import FGSM
from pgd import PGD



import functools
print = functools.partial(print, flush=True)


def to_list(t): return t.cpu().data.numpy().tolist()


def main():
    torch.backends.cudnn.benchmark = True

    parser = get_default_parser()
    config = parser.parse_args()

    make_dirs(config.ckpt_path)
    out = open(os.path.join(config.ckpt_path, "console.out"), "w")

    if config.dataset == "cifar10":
        train_loader, oe_loader, val_loader = cifar10(config)
        alpha=0.03131072223186493
        
        ds_mean = (0.4914, 0.4822, 0.4465) 
        ds_std = (0.2471, 0.2435, 0.2616) 

        
    elif config.dataset == "mnist":
        train_loader, oe_loader, val_loader = mnist(config)
        alpha=0.025456469506025314
        
        ds_mean = (0.1307, 0.1307, 0.1307)
        ds_std = (0.3081, 0.3081, 0.3081)

        
    elif config.dataset == "fashionmnist":
        train_loader, oe_loader, val_loader = fashionmnist(config)
        alpha=0.022218521684408188
        
        ds_mean =(0.2859, 0.2859, 0.2859)
        ds_std = (0.3530, 0.3530, 0.3530)


    elif config.dataset == "svhn":
        train_loader, oe_loader, val_loader = svhn(config)
        alpha=0.028310422553186493
        
        
        ds_mean =(0.491373, 0.482353, 0.446667)
        ds_std =(0.247059, 0.243529, 0.261569)

        

    elif config.dataset == "cifar100":
        train_loader, oe_loader, val_loader = cifar100(config)
        alpha=0.029434850439429283

        ds_mean =(0.491373, 0.482353, 0.446667)
        ds_std =(0.247059, 0.243529, 0.261569)


    elif config.dataset == "mvtec":
        config.batch_size=16
        train_loader, oe_loader, val_loader = mvtec(config)
        alpha=0.017563117668032646
        
        ds_mean=(0.485, 0.456, 0.406)
        ds_std=(0.229, 0.224, 0.225)
        
    else:
        raise NotImplementedError

    print(f"\nDataset {config.dataset} , Normal Class {config.normal_class}\n")
    
    mu = torch.tensor(ds_mean).view(3,1,1).cuda()
    std = torch.tensor(ds_std).view(3,1,1).cuda()
    
    
    normal_obj=normalizC(mu,std)
    
    save_model_info(config, file=out)

    f = resnet26(config, 1,normal_obj)
    f.cuda()

    if config.model == "adib":
        theta_0 = f.params()

    loss = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, f.parameters()),
                            lr=config.lr_sgd,
                            momentum=config.momentum_sgd,
                            weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                 milestones=list(
                                                     map(int, config.milestones.split(","))),
                                                 gamma=config.gamma)

    log = Log(file=out)
    log.register("time", format="{0:.4f}")
    log.register("loss", format="{0:.3f}")
    log.register("ap", format="{0:.3f}", color="yellow")
    log.register("auc", format="{0:.3f}", color="red")
    log.legend()

    # for epoch in range(config.num_epochs):
    
    num_epoches=0 if config.train=='False' else config.num_epochs
    
    
    # for epoch in range(num_epoches):
    for epoch in tqdm(range(num_epoches)):
        for i, batch in enumerate(zip(train_loader, oe_loader)):

            f.train()
            f.zero_grad()

            t = time.time()

            x = torch.cat((batch[0][0], batch[1][0]), 0)
            semi_targets = torch.cat((batch[0][1], batch[1][1]), 0)

            x, semi_targets = to_gpu(x, semi_targets)

            logits = f(x).squeeze()
            l = loss(logits, semi_targets.float())

            if config.model == "adib":
                l += config.alpha * \
                    torch.norm(f.params(backprop=True) - theta_0, 2)

            l.backward()
            optim.step()

            log.update("time", time.time() - t)
            log.update("loss", l.item(), x.size(0))
            log.report(which=["time", "loss"], epoch=epoch, batch_id=i)
            
        

        sched.step()
        print(f"Epoch {epoch}/{num_epoches} , loss : {l.item()}")
        
        if config.train=='True':
            # torch.save(f, './model.pth')
            torch.save(f, f'./model_ADIB_{config.dataset}_Class_{config.normal_class}.pth')
        
        
        # newline(f=out)

        # labels_scores = []

    
    
    mine_result = {}
    mine_result['Attack_Type'] = []
    mine_result['Attack_Target'] = []
    mine_result['ADV_AUC'] = []
    mine_result['setting'] = []
    
    
    if os.path.isfile(f'./model_ADIB_{config.dataset}_Class_{config.normal_class}.pth'):
        # f = torch.load('./model.pth')
        f = torch.load(f'./model_ADIB_{config.dataset}_Class_{config.normal_class}.pth')
        print("\nModel Loaded!\n")
    
    
    for att_type in ['fgsm', 'pgd']:
        
            
            print(f'\n\nAttack Type: {att_type}\n\n')

            
            clear_auc,normal_auc,anomal_auc,both_auc = testModel(f, val_loader, attack_type=att_type,alpha=alpha,epsilon=config.att_eps,just_clear=config.just_clear)

            mine_result['Attack_Type'].extend([att_type,att_type,att_type,att_type])
            mine_result['Attack_Target'].extend(['clean','normal','anomal','both'])
            mine_result['ADV_AUC'].extend([clear_auc,normal_auc,anomal_auc,both_auc])
            mine_result['setting'].extend([{'Dataset Name': config.dataset},{'Epsilon': config.att_eps},{'Alpha': alpha},{'Num Epoches': num_epoches}])
            
            print(f'Adv Adverserial Clean: {clear_auc}')
            print(f'Adv Adverserial Normal: {normal_auc}')
            print(f'Adv Adverserial Anomal: {anomal_auc}')
            print(f'Adv Adverserial Both: {both_auc}\n\n')

    df = pd.DataFrame(mine_result)    
    df.to_csv(os.path.join('./',f'Results_ADIB_{config.dataset}_Class_{config.normal_class}.csv'), index=False)


def testModel(f, val_loader, attack_type='fgsm',epsilon=8/255,alpha=0.01,just_clear=True):
   
    print(f"Attack Type : {attack_type} , epsilon : {epsilon} , alpha : {alpha}")
    
    
    labels_arr = []
    no_adv_scores_arr = []
    adv_scores_arr = []

    # for i, batch in enumerate(val_loader):
    for (i, batch) in enumerate(tqdm(val_loader, desc='Testing Adversarial')):
        
        x, labels = batch
        x, labels = to_gpu(x, labels)
        f.eval()
        
        no_adv_score=getScore(f,x)
        
        
        
        if just_clear=='False':
        
            if attack_type == 'fgsm':
                # adv_delta = fgsm(f, x, epsilon)
                # adv_delta = attack_pgd(f, x,epsilon= epsilon ,alpha= 1.25*epsilon ,attack_iters= 1)
                
                attack = FGSM(f, eps=epsilon)
                adv_images = attack(x,labels)

            if attack_type == 'pgd':
                # adv_delta = attack_pgd(f, x, epsilon=epsilon ,alpha= 2/255 ,attack_iters= 10)
                
                attack = PGD(f, eps=epsilon, alpha=alpha, steps=10, random_start=True)
                adv_images = attack(x, labels)
                

            # x = x+adv_delta if labels == 0 else x-adv_delta
            # x=torch.clamp(x, min=0, max=1)
            
            adv_scores=getScore(f,adv_images)
        else:
            adv_scores=no_adv_score
        
        no_adv_scores_arr.append(no_adv_score.detach().cpu().item())
        adv_scores_arr.append(adv_scores.detach().cpu().item())
        labels_arr.append(labels.detach().cpu().item())
    
    normal_imgs_idx=np.argwhere(np.array(labels_arr)==0).flatten().tolist()
    anomal_imgs_idx=np.argwhere(np.array(labels_arr)==1).flatten().tolist()
    
    
    clear_auc=roc_auc_score(labels_arr, no_adv_scores_arr)
    
    normal_auc=roc_auc_score(np.array(labels_arr)[normal_imgs_idx].tolist()+np.array(labels_arr)[anomal_imgs_idx].tolist(),np.array(adv_scores_arr)[normal_imgs_idx].tolist()+np.array(no_adv_scores_arr)[anomal_imgs_idx].tolist())
    
    anomal_auc=roc_auc_score(np.array(labels_arr)[normal_imgs_idx].tolist()+np.array(labels_arr)[anomal_imgs_idx].tolist(),np.array(no_adv_scores_arr)[normal_imgs_idx].tolist()+np.array(adv_scores_arr)[anomal_imgs_idx].tolist())
    
    both_auc=roc_auc_score(labels_arr, adv_scores_arr)
    
    
    return clear_auc,normal_auc,anomal_auc,both_auc

class normalizC:
    def __init__(self,mu,std) -> None:
        self.mu=mu 
        self.std=std
    def normalize(self,X):
        return (X - self.mu)/self.std

def getScore(f,x):
    scores = torch.sigmoid(f(x)).squeeze()
    return scores


if __name__ == "__main__":
    main()
