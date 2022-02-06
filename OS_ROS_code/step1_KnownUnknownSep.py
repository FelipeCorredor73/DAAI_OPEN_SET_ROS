
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss 


def _do_epoch(args,feature_extractor,rot_cls,obj_cls,cent_loss,source_loader,optimizer,device):

    # Set up
    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    # Useful variables and initializations
    k = len(source_loader.dataset) # Number of images
    cls_corrects, rot_corrects  = 0, 0

    # You pass all images and labels from the data loader through the training 
    # process, the number of iterations is defined by the number of images you 
    # pass each iteration (Batch size)
    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):

        # Take data to CUDA
        data, class_l, data_rot, rot_l  = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)
        
        # Gradients at zero
        optimizer.zero_grad()

        # Feed the feature extractor (in train mode) with the input data
        features = feature_extractor(data)
        features_rot = feature_extractor(data_rot)
        
        # Feed the created networks (in train mode) with class and rotation data
        # to make correct data inputs
        y_cls = obj_cls(features)

        # Concatenate features of rotated and original images
        features_rot = torch.cat((features, features_rot),1)
        y_rot   = rot_cls(features_rot)

        # ----------------------------------------------------------------------
        # Loss calculation
        # Softmax is done within CrossEntropyLoss definition
        class_loss  = criterion(y_cls, class_l)
        rot_loss    = criterion(y_rot, rot_l)
        cnt_loss    = (cent_loss(features, class_l))

        loss = class_loss + args.weight_RotTask_step1*rot_loss + args.weight_Center_Loss*cnt_loss
        loss.backward()

        # Following the advice of the center loss git we remove the effect of the
        # hyperparameter on the adjustment of the centers, but we assume the same
        # learning rate is used for both networks and center loss (lr_cent = lr)
        if (args.weight_Center_Loss > 0):
            for param in cent_loss.parameters():
                param.grad.data *= (1/(args.weight_Center_Loss))

        optimizer.step()

        # Predictions
        cls_pred = nn.Softmax(dim=-1)(y_cls)
        rot_pred = nn.Softmax(dim=-1)(y_rot)

        _, cls_pred = torch.max(cls_pred, dim=1)
        _, rot_pred = torch.max(rot_pred, dim=1)

        # Correct classifications 
        cls_corrects += torch.sum(cls_pred == class_l)
        rot_corrects += torch.sum(rot_pred == rot_l)

    print('   Correct class predictions %d' % cls_corrects,'of',float(k))
    print('   Correct rot predictions %d' % rot_corrects,'of',float(k))
    
    # Accuracy calculation
    acc_cls = cls_corrects/float(k)
    acc_cls = acc_cls*100
    acc_rot = rot_corrects/float(k)
    acc_rot = acc_rot*100

    return class_loss, acc_cls, rot_loss, acc_rot, cnt_loss


def step1(args,feature_extractor,rot_cls,obj_cls,source_loader,device):
    cent_loss = CenterLoss(num_classes = args.n_classes_known, feat_dim = 512, use_gpu=True)
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, cent_loss, args.weight_Center_Loss, args.epochs_step1, args.learning_rate, args.train_all)

    for epoch in range(args.epochs_step1):
        print('   Epoch: ',epoch)
        class_loss, acc_cls, rot_loss, acc_rot, cnt_loss = _do_epoch(args,feature_extractor,rot_cls,obj_cls,cent_loss,source_loader,optimizer,device)
        print("   Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f, Center-Loss %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot, cnt_loss.item()))
        scheduler.step()


