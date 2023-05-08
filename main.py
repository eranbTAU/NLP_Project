import argparse
import torch
# from Model import CNN, train
# from dataloader import data_loaders, save_net



parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--root_train', default=r'/home/eranbamani/Documents/data_PointProject/MarkerData_Reg/train', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('--saveM_path', default=r'checkpoint', metavar='DIR',
                    help='path for save the weights in optimizer of the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--criterion', default=r'rmse', metavar='CRI',
                    help='Criterion loss. (default: rmse)')

# # Optimizer parameters
parser.add_argument('--optim', type=str, default='SGD',
                    help='define optimizer type')
parser.add_argument('--scheduler', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1.7414537048734527e-05, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 2)')

# # Misc
parser.add_argument('--img_size', type=int, default=224, metavar='Size',
                    help='Image size for resize')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log_wandb', action='store_true', default=True,
                    help='log training and validation metrics to wandb')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='type "cpu" if there is no gpu')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument("--load_model", default=False, type=str)


def main(args_config, checkpoint=None, trans_learn=False):
    size = args_config.img_size
    train_path = args_config.root_train

    train_dataloader, val_dataloader = data_loaders(train_path, size)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    model = CNN()
    model = model.to(device)
    if trans_learn:
        model.load_state_dict(torch.load(checkpoint, map_location=args_config.device))

    try:
        train(model, train_dataloader, val_dataloader, args_config.epochs, args_config.lr, args_config.device)
        PATH = r'/home/eranbamani/Documents/weights/Estimator'
        save_net(PATH, model.state_dict())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args_config = parser.parse_args()
    # main(args_config)


