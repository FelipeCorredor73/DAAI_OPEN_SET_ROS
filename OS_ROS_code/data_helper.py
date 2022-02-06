
import torch
from torchvision import transforms

from dataset import Dataset, TestDataset, _dataset_info


def get_train_dataloader(args,txt_file):


    img_transformer = get_train_transformers(args)
    name_train, labels_train = _dataset_info(txt_file)
    train_dataset = Dataset(name_train, labels_train, args.path_dataset, img_transformer=img_transformer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return loader


def get_val_dataloader(args,txt_file):

    names, labels = _dataset_info(txt_file)
    img_tr = get_test_transformer(args)
    test_dataset = TestDataset(names, labels,args.path_dataset, img_transformer=img_tr)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader


def get_train_transformers(args):

    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]

    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    if args.random_grayscale:
        img_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


def get_test_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)