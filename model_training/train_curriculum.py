# do random seeding
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FinCLData
import argparse
from model import *
import pickle
import torch.optim as optim
import copy
import time
import os
import random
from sklearn.metrics import matthews_corrcoef, accuracy_score
from torch.utils.data import Dataset


start_time = time.strftime("%Y%m%d-%H%M%S")
print(start_time)
parser = argparse.ArgumentParser(description="FinNLPCL Model")

parser.add_argument(
    "--model",
    default="time",
    type=str,
    help="Model to use for training [simple, time] (default: time)",
)

parser.add_argument(
    "--lr",
    default=0.0003,
    type=float,
    help="Learning rate to use for training (default: 0.0003)",
)

parser.add_argument(
    "--data",
    default="stock",
    help="data to be used [stock, china] (default: stock)",
)

parser.add_argument(
    "--num_epochs",
    default=500,
    type=int,
    help="Number of epochs to run for training (default: 500)",
)

parser.add_argument(
    "--num_buckets",
    default=10,
    type=int,
    help="Number of epochs to run for training (default: 500)",
)

parser.add_argument(
    "--epochs_per_bucket",
    default=5,
    type=int,
    help="Number of epochs to run for training on each bucket (default: 5)",
)
parser.add_argument(
    "--seed",
    default=2020,
    type=int,
    help="Seed for experiment (default: 2020)",
)
parser.add_argument(
    "--decay",
    default=1e-5,
    type=float,
    help="Weight decay to use for training (default: 1e-5)",
)

parser.add_argument(
    "--alpha",
    default=0.5,
    type=float,
    help="Weight of text_price complexity",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Batch Size use for training the model (default: 32)",
)
parser.add_argument(
    "--task",
    default="movement",
    help="Task on which to train model [movement, volatility] (default: movement)",
)

parser.add_argument(
    "--use_attn1",
    default=True,
    type=bool,
    help="Whether to use attention in LSTM1 (default: True)",
)
parser.add_argument(
    "--use_attn2",
    default=True,
    type=bool,
    help="Whether to use attention in LSTM2 (default: True)",
)

parser.add_argument(
    "--name",
    default="with_curr",
    type=str,
    help="exp_name",
)


args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# not doing seeding on CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def min_max_scale(x):
    min_x = np.min(x)
    max_x = np.max(x)
    x = (x - min_x) / (max_x - min_x)
    return x


if args.data == "stock":
    # Load the difficulty scores
    with open("Enter path to text_price_difficulty_stocknet.pkl here", "rb") as f:
        text_price_difficulty = min_max_scale(np.array(pickle.load(f)))

    with open("Enter path to model_difficulty.pkl here", "rb") as f:
        model_difficulty = min_max_scale(np.array(pickle.load(f)))

elif args.data == "china":
    with open(
        "Enter path to text_price_difficulty_chinese.pkl here",
        "rb",
    ) as f:
        text_price_difficulty = min_max_scale(np.array(pickle.load(f)))

    with open(
        "Enter path to model_difficulty_chinese.pkl here",
        "rb",
    ) as f:
        model_difficulty = min_max_scale(np.array(pickle.load(f)))


if args.task == "movement":
    criterion = nn.CrossEntropyLoss()
elif args.task == "volatility":
    criterion = nn.MSELoss()

if args.task == "movement":
    outdim = 2
elif args.task == "volatility":
    outdim = 1


class CLData(Dataset):
    """"""

    def __init__(self, data):
        """
        data: the data
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        return temp


if os.path.exists("saved_models") == False:
    os.mkdir("saved_models")

loss_history = {"train": [], "val": []}
accuracy_history = {"train": [], "val": []}
mcc_history = {"train": [], "val": []}

since = time.time()
best_loss = 9999999
best_epoch = 0
best_wt_price = 0.0
best_mcc = 0
best_acc = 0

buckets = args.num_buckets


alpha_list = np.linspace(0, 1, 21)

for alpha in alpha_list:


    # Calculate the weighted difficulty
    total_difficulty = alpha * text_price_difficulty + (1 - alpha) * model_difficulty
    # sort and define the curriculum
    curriculum = np.argsort(total_difficulty)

    if args.data == "stock":
        with open(
            "Enter path to train_data_stocknet.pkl here",
            "rb",
        ) as f:
            traindata = pickle.load(f)
    elif args.data == "china":
        with open(
            "Enter path to train_data_chinese.pkl here",
            "rb",
        ) as f:
            traindata = pickle.load(f)

    train_data_sorted = [traindata[x] for x in curriculum]
    train_data_sorted = CLData(train_data_sorted)

    data_in_each = int(len(train_data_sorted) / buckets)
    number_in_last = len(train_data_sorted) - ((buckets - 1) * data_in_each)
    train_buckets = [
        train_data_sorted[i * data_in_each : (i + 1) * data_in_each]
        for i in range(0, buckets - 1)
    ]
    train_buckets.append(
        train_data_sorted[
            (buckets - 1) * data_in_each : (buckets - 1) * data_in_each + number_in_last
        ]
    )

    if args.model == "simple":
        if args.data == "stock":
            model = FinNLPCL(
                text_embed_dim=768,
                intraday_hiddenDim=256,
                interday_hiddenDim=256,
                intraday_numLayers=1,
                interday_numLayers=1,
                use_attn1=args.use_attn1,
                use_attn2=args.use_attn2,
                maxlen=30,
                outdim=outdim,
                device=device,
            )
        elif args.data == "china":
            model = FinNLPCL(
                text_embed_dim=768,
                intraday_hiddenDim=256,
                interday_hiddenDim=256,
                intraday_numLayers=1,
                interday_numLayers=1,
                use_attn1=args.use_attn1,
                use_attn2=args.use_attn2,
                maxlen=10,
                outdim=outdim,
                device=device,
            )

    elif args.model == "time":
        if args.data == "stock":
            model = TimeFinNLPCL(
                text_embed_dim=768,
                intraday_hiddenDim=256,
                interday_hiddenDim=256,
                intraday_numLayers=1,
                interday_numLayers=1,
                use_attn1=args.use_attn1,
                use_attn2=args.use_attn2,
                maxlen=30,
                outdim=outdim,
                device=device,
            )
        elif args.data == "china":
            model = TimeFinNLPCL(
                text_embed_dim=768,
                intraday_hiddenDim=256,
                interday_hiddenDim=256,
                intraday_numLayers=1,
                interday_numLayers=1,
                use_attn1=args.use_attn1,
                use_attn2=args.use_attn2,
                maxlen=10,
                outdim=outdim,
                device=device,
            )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model.train()

    # the curriculum thing
    phase = "train"

    for bucket in range(buckets):
        print("Bucket {}/{}".format(bucket, buckets - 1))
        print("-" * 10)

        data_loader_bucket = torch.utils.data.DataLoader(
            train_buckets[bucket],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
        )

        for epoch in range(args.epochs_per_bucket):
            print("Epoch {}/{}".format(epoch, args.epochs_per_bucket - 1))
            print("-" * 10)

            running_loss = 0.0

            truelabels = []
            predlabels = []

            for batch_data in data_loader_bucket:
                embedding_data = batch_data["embedding"]
                embedding_data = embedding_data.to(device)

                if args.task == "movement":
                    truelabels.extend(batch_data["movement_label"].numpy())
                    target = batch_data["movement_label"]
                    target = target.type(torch.LongTensor).to(device)
                elif args.task == "volatility":
                    target = batch_data["volatility"]
                    target[torch.isnan(target)] = 0
                    target[torch.isinf(target)] = 0
                    target = target.type(torch.FloatTensor).to(device).unsqueeze(-1)

                length = batch_data["length_data"]

                if args.model == "time":
                    time_feats = batch_data["time_feature"].to(device).squeeze(-1)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # print(embedding_data)
                    if args.model == "simple":
                        outputs = model(embedding_data, length)
                    elif args.model == "time":
                        outputs = model(embedding_data, length, time_feats)

                    loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * target.size(0)
                if args.task == "movement":
                    predlabels.extend(
                        torch.argmax(outputs, dim=-1).cpu().detach().numpy()
                    )

            if args.task == "movement":
                predlabels = np.array(predlabels)
                truelabels = np.array(truelabels)

            epoch_loss = running_loss / len(data_loader_bucket.dataset)
            loss_history[phase].append(epoch_loss)

            if args.task == "movement":
                epoch_accuracy = accuracy_score(truelabels, predlabels)
                epoch_mcc = matthews_corrcoef(truelabels, predlabels)
                accuracy_history[phase].append(epoch_accuracy)
                mcc_history[phase].append(epoch_mcc)

                print(
                    "{} Epoch: {} Acc: {:.4f} MCC: {:.4f} Loss: {:.4f}".format(
                        phase, epoch, epoch_accuracy, epoch_mcc, epoch_loss
                    )
                )
            elif args.task == "volatility":
                print("{} Epoch: {} Loss: {:.4f}".format(phase, epoch, epoch_loss))

        print()

    if args.data == "stock":
        traindata = FinCLData("Enter path to train_data_stocknet.pkl here")
        valdata = FinCLData("Enter path to val_data_stocknet.pkl here")
    elif args.data == "china":
        traindata = FinCLData("Enter path to train_data_chinese.pkl here")
        valdata = FinCLData("Enter path to val_data_chinese.pkl here")

    # The normal training (after the curriculum thing)
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    dataloaders = {"train": trainloader, "val": valloader}

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            truelabels = []
            predlabels = []
            # Iterate over data.
            for batch_data in dataloaders[phase]:
                embedding_data = batch_data["embedding"]
                embedding_data = embedding_data.to(device)

                if args.task == "movement":
                    truelabels.extend(batch_data["movement_label"].numpy())
                    target = batch_data["movement_label"]
                    target = target.type(torch.LongTensor).to(device)
                elif args.task == "volatility":
                    target = batch_data["volatility"]
                    target[torch.isnan(target)] = 0
                    target[torch.isinf(target)] = 0
                    target = target.type(torch.FloatTensor).to(device).unsqueeze(-1)

                length = batch_data["length_data"]

                if args.model == "time":
                    time_feats = batch_data["time_feature"].to(device).squeeze(-1)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # print(embedding_data)
                    if args.model == "simple":
                        outputs = model(embedding_data, length)
                    elif args.model == "time":
                        outputs = model(embedding_data, length, time_feats)

                    loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * target.size(0)

                if args.task == "movement":
                    predlabels.extend(
                        torch.argmax(outputs, dim=-1).cpu().detach().numpy()
                    )

            if args.task == "movement":
                predlabels = np.array(predlabels)
                truelabels = np.array(truelabels)

            epoch_loss_total = running_loss / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss_total)

            if args.task == "movement":

                epoch_accuracy_total = accuracy_score(truelabels, predlabels)
                epoch_mcc_total = matthews_corrcoef(truelabels, predlabels)
                accuracy_history[phase].append(epoch_accuracy_total)
                mcc_history[phase].append(epoch_mcc_total)

                if phase == "val" and epoch_accuracy_total > best_acc:
                    best_acc = epoch_accuracy_total
                    best_mcc = epoch_mcc_total
                    best_loss = epoch_loss_total
                    best_alpha = alpha
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == "val":
                    torch.save(
                        {
                            "model_wts": model.state_dict(),
                            "current_epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_loss": best_loss,
                            "best_accuracy": best_acc,
                            "best_mcc": best_mcc,
                            "loss_history": loss_history,
                            "mcc_history": mcc_history,
                            "accuracy_history": accuracy_history,
                            "best_model_wts": best_model_wts,
                            "best_alpha": best_alpha,
                            "args": args,
                        },
                        "saved_models/"
                        + str(args.name)
                        + "_"
                        + str(args.data)
                        + "_"
                        + str(args.num_epochs)
                        + "_"
                        + str(start_time)
                        + ".pth",
                    )

                    print(
                        "Epoch: {} Alpha: {:.4f} Best Acc: {:.4f}  MCC: {:.4f}  Val Loss: {:.4f}".format(
                            epoch,
                            best_alpha,
                            best_acc,
                            best_mcc,
                            best_loss,
                        )
                    )

                print(
                    "{} Epoch: {} Alpha: {:.4f} Acc: {:.4f} MCC: {:.4f} Loss: {:.4f}".format(
                        phase,
                        epoch,
                        alpha,
                        epoch_accuracy_total,
                        epoch_mcc_total,
                        epoch_loss_total,
                    )
                )
            elif args.task == "volatility":

                if phase == "val" and epoch_loss_total < best_loss:
                    best_loss = epoch_loss_total
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_alpha = alpha

                if phase == "val":
                    torch.save(
                        {
                            "model_wts": model.state_dict(),
                            "current_epoch": epoch,
                            "best_epoch": best_epoch,
                            "best_loss": best_loss,
                            "loss_history": loss_history,
                            "best_model_wts": best_model_wts,
                            "best_alpha": best_alpha,
                            "args": args,
                        },
                        "saved_models/"
                        + str(args.name)
                        + "_"
                        + str(args.data)
                        + "_"
                        + str(args.num_epochs)
                        + "_"
                        + str(start_time)
                        + ".pth",
                    )
                    print(
                        "Epoch: {} Alpha: {:.4f} Best Val Loss: {:.4f}  ".format(
                            epoch, best_alpha, best_loss
                        )
                    )

                print(
                    "{} Epoch: {}  Alpha: {:.4f} Loss: {:.4f} ".format(
                        phase,
                        epoch,
                        alpha,
                        epoch_loss_total,
                    )
                )

        print(start_time)
        print()

time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
)
print("Best val Loss: {:4f}".format(best_loss))
