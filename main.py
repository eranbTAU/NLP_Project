import argparse
import torch
from utils import path2data
from Models import PreTrmodel
from dataloader import data_loaders
import torch
import torch.nn.utils.prune as prune

# Config the parser
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('--data_root', default=r'./data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data_name', default='ireland', metavar='DIR',
                    help='name to the require dataset')
parser.add_argument('--saveM_path', default=r'checkpoint', metavar='DIR',
                    help='path for save the weights in optimizer of the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--criterion', default=r'rmse', metavar='CRI',
                    help='Criterion loss. (default: rmse)')
parser.add_argument('--model_name', type=str, default='albert-base-v2', metavar='N',
                    help='The name of the model')
# # Optimizer parameters
parser.add_argument('--optim', type=str, default='SGD',
                    help='define optimizer type')
parser.add_argument('--scheduler', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1.7414537048734527e-05, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--alpha', type=int, default=0.9473, metavar='N',
                    help='alpha parameter for optimizer')
parser.add_argument('--beta', type=int, default=0.962, metavar='N',
                    help='beta parameter for optimizer')
parser.add_argument('--weight_decay', type=int, default=0.0087, metavar='N',
                    help='weight_decay parameter for optimizer')
# # Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 0)')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='type "cpu" if there is no gpu')
parser.add_argument("--drop_last", default=True, type=str)
parser.add_argument("--load_model", default=False, type=str)


def main(args_config, checkpoint=None, trans_learn=False):

    path = path2data(args_config)
    train_data, val_data, test_data = data_loaders(args_config, path)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    model = PreTrmodel(args_config)

    model = model.to(device)

    for param in list(model.named_parameters()):
        prune.l1_unstructured(model.albert.embeddings.word_embeddings, name="weight", amount=3)

    for buffer in model.named_buffers():
        print(buffer)

    optimizer = torch.optim.Adam(model.parameters(), lr=args_config.lr, betas=(args_config.alpha, args_config.beta),
                                 weight_decay=args_config.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    # try:
    #     train(model, train_dataloader, val_dataloader, args_config.epochs, args_config.lr, args_config.device)
    #     PATH = r'/home/eranbamani/Documents/weights/Estimator'
    #     save_net(PATH, model.state_dict())
    # except KeyboardInterrupt:
    #     pass


if __name__ == '__main__':
    args_config = parser.parse_args()
    main(args_config)


