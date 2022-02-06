
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss
from itertools import cycle
import numpy as np

#### Implement Step2

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    k_source = len(source_loader.dataset)       # Number of images of source dataset
    k_target = len(target_loader_train.dataset) # Number of images of train target dataset
    print(k_source), print(k_target)

    cls_corrects, rot_corrects  = 0, 0

    target_loader_train = cycle(target_loader_train)

    k_rot =  float(k_source)/float(k_target)
    k_rot = k_target*k_rot

    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):

        # With this method we can do a batch of the source and a batch of the 
        # target at the same time, not iterating one over the other one
        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        optimizer.zero_grad()

        features_s = feature_extractor(data_source)
        features_t = feature_extractor(data_target)
        features_t_rot = feature_extractor(data_target_rot)
        features_t_rot = torch.cat((features_t,features_t_rot),1)

        # Train the classifier with the target images to train the unknown class
        y_t_cls = obj_cls(features_s)
        y_t_rot = rot_cls(features_t_rot)

        class_loss = criterion(y_t_cls, class_l_source)
        rot_loss = criterion(y_t_rot, rot_l_target)

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        loss.backward()

        optimizer.step()

        cls_pred = nn.Softmax(dim=-1)(y_t_cls)
        rot_pred = nn.Softmax(dim=-1)(y_t_rot)

        _, cls_pred = torch.max(y_t_cls, dim=1)
        _, rot_pred = torch.max(y_t_rot, dim=1)

        cls_corrects += torch.sum(cls_pred == class_l_source)
        rot_corrects += torch.sum(rot_pred == rot_l_target)

    # Accuracy calculation
    acc_cls = float(cls_corrects)/float(k_source)
    acc_cls = acc_cls*100
    acc_rot = float(rot_corrects)/float(k_source)
    acc_rot = acc_rot*100

    print('   Correct class predictions %d' % cls_corrects,'of',float(k_source))
    print('   Correct rot predictions %d' % rot_corrects,'of',float(k_target))

    print("   Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))

    print('-------------------- Evaluation of the network --------------------')
    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    cls_eval_known = 0    # Number of correct known predictions
    cls_eval_unknown = 0  # Number of correct unknown predictions
    k_known = 0
    k_unknown = 0
    k_mistakes = 0
    k_eval_t = len(target_loader_eval)

    with torch.no_grad():
        for it, (data, class_l,_,_) in enumerate(target_loader_eval):
          data, class_l = data.to(device), class_l.to(device)

          features = feature_extractor(data)
          y_cls = obj_cls(features)
          y_cls = nn.Softmax(dim=-1)(y_cls)
          
          _, y_cls = torch.max(y_cls, dim=1)

          # Correctly predicted labeled
          if (y_cls == class_l):
              cls_eval_known += torch.sum((class_l<args.n_classes_known)).data.item()     # Using known labels
              cls_eval_unknown += torch.sum((class_l>=args.n_classes_known)).data.item()  # Using unknown label
          else:
              k_mistakes += 1

          k_known += torch.sum(class_l<args.n_classes_known).data.item()
          k_unknown += torch.sum(class_l>=args.n_classes_known).data.item()

    if (k_unknown>0):
        UNK = (cls_eval_unknown/float(k_unknown))
    else:
        UNK = cls_eval_unknown
    OS_star = (cls_eval_known/float(k_known))
    HOS = 2*(OS_star*UNK)/(OS_star + UNK)
    cls_eval = cls_eval_known + cls_eval_unknown

    print('   Correct class predictions %d' % cls_eval,'of', str(k_eval_t))
    print('   Known classes correct predictions %d' % cls_eval_known, 'of', str(k_known))
    print('   Unknown class correct predictions %d' % cls_eval_unknown, 'of', str(k_unknown))
    print("   Open Set metrics: ")
    print("   OS*: ", OS_star*100)
    print("   UNK: ", UNK*100)
    print("   HOS: ", HOS*100)
    print("   Mistakes:", k_mistakes)


def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    cent_loss = CenterLoss(num_classes = args.n_classes_known, feat_dim = 512, use_gpu=True)
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, cent_loss, 0, args.epochs_step2, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step2):
        print("Epoch: ", epoch, "----------------------------------------")
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        scheduler.step()
