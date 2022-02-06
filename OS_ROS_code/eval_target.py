
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import tensorflow as tf
import os


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()

    k = len(target_loader_eval.dataset) # Number of images
    normality_scorei = torch.tensor([0],device=device)

    EntropyScore = torch.zeros((k,4), device=device)
    RotScore = torch.zeros((k,4), device=device)

    ground_truth = torch.zeros((k,), device=device)
    Pred_UK = torch.zeros((k,), device=device)
    normality_score = torch.zeros((k,), device=device)

    # Call the number of known classes constant, take its log and make it a tensor
    logCS = (torch.as_tensor(np.log(args.n_classes_known),dtype=torch.float32)).to(device)

    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

            features = feature_extractor(data)
            # -----------------------------------------------------------------%
            # Explore all possible rotations ----------------------------------%
            # -----------------------------------------------------------------%

            Zi_r = torch.zeros((1,4), device=device)
            entropy_i = torch.zeros((1,4), device=device)
            Zi_r = torch.as_tensor(Zi_r, dtype=torch.float32, device=device)
            entropy_i = torch.as_tensor(entropy_i, dtype=torch.float32, device=device)
            for rot_it in range(4):
                # Compute rotation rot_it
                data_rot = torch.rot90(data,k=rot_it,dims=[2,3])
                rot_l = rot_it

                # Pass rotation
                features_rot = feature_extractor(data_rot)
                rot_pred = rot_cls(torch.cat((features,features_rot),1))
                # Prediction
                Zi = torch.nn.Softmax(dim=-1)(rot_pred)           # p_rot = Zi
                entropy = (Zi)*torch.log((Zi) + 1e-21)/logCS      # entropy = H(Zi)
   
                Zi_r += Zi
                entropy_i += entropy

            # -----------------------------------------------------------------%
            # Managing data from all rotations --------------------------------%
            # -----------------------------------------------------------------%

            RotScore[it] = Zi_r
            EntropyScore[it] = (entropy_i/4)

            if (class_l >= args.n_classes_known):
                ground_truth[it] = 0
            else:
                ground_truth[it] = 1

    # Computing Normality Score
    RS_std,RS_mean = torch.std_mean(RotScore)
    RotScore = (RotScore - RS_mean)/RS_std
    EntropyScore = 1 - EntropyScore
    ES_std,ES_mean = torch.std_mean(EntropyScore)
    EntropyScore = (EntropyScore - ES_mean)/ES_std
    normality_score = torch.maximum(RotScore,EntropyScore)
    normality_score = torch.mean(normality_score,dim=1)
    NS_std,NS_mean = torch.std_mean(normality_score)
    normality_score = (normality_score - NS_mean)/NS_std

    # -------------------------------------------------------------------------%
    # Separate known from unknown and calculate the AUROC score ---------------%
    # -------------------------------------------------------------------------%

    for it1 in range(len(normality_score)):
        if (normality_score[it1] > args.threshold):
            Pred_UK[it1] = 1
        else:
            Pred_UK[it1] = 0

    normality_score_cpu = normality_score.cpu()
    ground_truth_cpu = ground_truth.cpu()
    normality_score_cpu = np.array(normality_score_cpu)
    ground_truth_cpu = np.array(ground_truth_cpu)

    auroc = roc_auc_score(ground_truth_cpu,normality_score_cpu)
    print('   AUROC %.4f' % auroc)

    # -------------------------------------------------------------------------%
    # Update the known and unknown samples at the source and target files -----%
    # -------------------------------------------------------------------------%
    number_of_known_samples = (torch.sum(Pred_UK)).int()
    number_of_unknown_samples = (k - torch.sum(Pred_UK)).int()

    print('   The number of target samples selected as known is: ', number_of_known_samples,' of ',torch.sum((ground_truth),0).int())
    print('   The number of target samples selected as unknown is: ', number_of_unknown_samples, ' of ', (k - (torch.sum((ground_truth),0))).int())

    images_labels = target_loader_eval.dataset.labels 
    images_paths = target_loader_eval.dataset.names

    known_images_paths, known_images_labels, unknown_images_paths = [], [], []

    # Known/unknown paths and labels for evaluated images
    for it2 in range(k):
        if (Pred_UK[it2]==1):
            known_images_labels = known_images_labels+[images_labels[it2]]
            known_images_paths = known_images_paths+[images_paths[it2]]
        else:
            unknown_images_paths = unknown_images_paths+[images_paths[it2]]

    # Create new txt files
    rand = random.randint(0,100000)
    rand = 260
    print('   Generated random number is :', rand)
    if not os.path.isdir('/content/drive/MyDrive/DAAI21/ROS_AMLProject/new_txt_list/'):
        os.mkdir('/content/drive/MyDrive/DAAI21/ROS_AMLProject/new_txt_list/')

    # This txt files will have names of source images and names of target images
    # selected as unknown. [Ds_known + Dt_unknown]
    target_unknown = open('new_txt_list/'+args.source+'_unknown_'+str(rand)+'.txt','w')
    for it3 in range (number_of_unknown_samples):
        target_unknown.write(unknown_images_paths[it3]+' '+str(args.n_classes_known)+'\n')
        # Label will always be 45, the unknown class label

    files = ['txt_list/'+args.source+'_known'+'.txt','new_txt_list/'+args.source+'_unknown_'+str(rand)+'.txt']
    with open('new_txt_list/'+args.source+'_known_'+str(rand)+'.txt', 'w') as outfile:
        for names in files:
            with open(names) as infile:
                # read the data from Ds and Dt_unknown and write [Ds U Dt_unknown]
                outfile.write(infile.read())

    # This txt files will have the names of the target images selected as known
    # [Dt_known]
    target_known = open('new_txt_list/'+args.target+'_known_'+str(rand)+'.txt','w')
    for it4 in range (number_of_known_samples):
        target_known.write(known_images_paths[it4]+' '+str(known_images_labels[it4])+'\n')

    return rand
