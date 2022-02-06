
import argparse

import torch

import data_helper
from resnet import resnet18_feat_extractor, Classifier

from step1_KnownUnknownSep import step1
from eval_target import evaluation
from step2_SourceTargetAdapt import step2


from torch import nn
from optimizer_helper import get_optim_and_scheduler


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", default='Art', help="Source name")
    parser.add_argument("--target", default='Clipart', help="Target name")
    parser.add_argument("--n_classes_known", type=int, default=45, help="Number of known classes")
    parser.add_argument("--n_classes_tot", type=int, default=65, help="Number of unknown classes")

    # dataset path
    parser.add_argument("--path_dataset", default="/content/drive/MyDrive/DAAI21/ROS_AMLProject/data/", help="Path where the Office-Home dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    parser.add_argument("--epochs_step1", type=int, default=10, help="Number of epochs of step1 for known/unknown separation")
    parser.add_argument("--epochs_step2", type=int, default=10, help="Number of epochs of step2 for source-target adaptation")

    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    # Parameters to play (hyperparameters)
    parser.add_argument("--weight_RotTask_step1", type=float, default=0.5, help="Weight for the rotation loss in step1")
    parser.add_argument("--weight_RotTask_step2", type=float, default=0.5, help="Weight for the rotation loss in step2")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for the known/unkown separation") 
    parser.add_argument("--weight_Center_Loss", type=float, default=0.5, help="Weight for the Center Loss in step1")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    print("Args obtained")

    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the network with a number of classes equals to the number of known classes + 1 
        # (the unknown class, trained only in step2)
        self.feature_extractor = resnet18_feat_extractor()
        self.obj_classifier = Classifier(512,self.args.n_classes_known+1)
        self.rot_classifier = Classifier(512*2,4)

        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj_cls = self.obj_classifier.to(self.device)
        self.rot_cls = self.rot_classifier.to(self.device)

        source_path_file = 'txt_list/'+args.source+'_known.txt'
        self.source_loader = data_helper.get_train_dataloader(args,source_path_file)

        target_path_file = 'txt_list/'+args.target+'.txt'
        self.target_loader_train = data_helper.get_val_dataloader(args,target_path_file)
        self.target_loader_eval = data_helper.get_val_dataloader(args,target_path_file)

        Openness = 1 - self.args.n_classes_known/self.args.n_classes_tot

        print("Source: ",self.args.source," Target: ",self.args.target)
        print("Dataset size: source %d, target %d" % (len(self.source_loader.dataset), len(self.target_loader_train.dataset)))
        print("Batch size: ", self.args.batch_size)
        print("Openness: %f" % Openness)
        print("Rotation task 1 at step 1: %f" % self.args.weight_RotTask_step1)
        print("Rotation task 2 at step 1: %f" % self.args.weight_RotTask_step2)
        print("Known/Unknown separation threshold: %f" % self.args.threshold)
        print("Learning rate: %f" % self.args.learning_rate)


    def do_training(self):

        print('+ Step 1 --------------------------------------------')
        #step1(self.args,self.feature_extractor,self.rot_cls,self.obj_cls,self.source_loader,self.device)

        print('+ Target - Evaluation -- for known/unknown separation')
        rand = evaluation(self.args,self.feature_extractor,self.rot_cls,self.target_loader_eval,self.device)
        
        # new dataloaders
        source_path_file = 'new_txt_list/' + self.args.source + '_known_'+str(rand)+'.txt'
        self.source_loader = data_helper.get_train_dataloader(self.args,source_path_file)

        target_path_file = 'new_txt_list/' + self.args.target + '_known_' + str(rand) + '.txt'
        self.target_loader_train = data_helper.get_train_dataloader(self.args,target_path_file)
        self.target_loader_eval = data_helper.get_val_dataloader(self.args,target_path_file)

        print('+ Step 2 --------------------------------------------')
        step2(self.args,self.feature_extractor,self.rot_cls,self.obj_cls,self.source_loader,self.target_loader_train,self.target_loader_eval,self.device)


def main():
    args = get_args()
    trainer = Trainer(args)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()


