# do random seeding
import torch
import torch.nn as nn
import random
import numpy as np
from dataset import FinCLData
import argparse
from model import *
import torch.optim as optim
import copy
import time
import os
import random
from sklearn.metrics import matthews_corrcoef, accuracy_score


start_time = time.strftime("%Y%m%d-%H%M%S")
print(start_time)
parser = argparse.ArgumentParser(description="FinNLPCL Model")

parser.add_argument(
    "--model",
    default="time",
    type=str,
    help="Model to use for training [simple, time] (default: simple)",
)

parser.add_argument(
    "--lr",
    default=0.0003,
    type=float,
    help="Learning rate to use for training (default: 0.0003)",
)
parser.add_argument(
    "--num_epochs",
    default=500,
    type=int,
    help="Number of epochs to run for training (default: 500)",
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
    "--batch_size",
    default=32,
    type=int,
    help="Batch Size use for training the model (default: 32)",
)

parser.add_argument(
    "--data",
    default="stock",
    help="data to be used [stock, china] (default: stock)",
)

parser.add_argument(
    "--bucket",
    default=0,
    help="Bucket number to train on",
)

parser.add_argument(
    "--use_attn1",
    default=True,
    type=bool,
    help="Whether to use attention in LSTM1 (default: False)",
)
parser.add_argument(
    "--use_attn2",
    default=True,
    type=bool,
    help="Whether to use attention in LSTM2 (default: False)",
)



args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# not doing seeding on CUDA
device = torch.device("cuda")

if args.data == "stock":
    traindata = FinCLData("Enter path to train_data_stocknet.pkl here")
    valdata = FinCLData("Enter path to test_data_stocknet.pkl here")
elif args.data == "china":
    traindata = FinCLData("Enter path to train_data_chinese.pkl here")
    valdata = FinCLData("Enter path to test_data_chinese.pkl here")

buckets = 10
data_in_each = int(len(traindata)/buckets)
number_in_last = len(traindata)- ((buckets-1)*data_in_each)

train_buckets = [traindata[i*data_in_each:(i+1)*data_in_each] for i in range(0,buckets-1)]
train_buckets.append(traindata[(buckets-1)*data_in_each : (buckets-1)*data_in_each+number_in_last])
bucket_number = int(args.bucket)


trainloader = torch.utils.data.DataLoader(
    train_buckets[bucket_number], batch_size=args.batch_size, shuffle=True, num_workers=8
)
valloader = torch.utils.data.DataLoader(
    valdata, batch_size=args.batch_size, shuffle=True, num_workers=8
)

dataloaders = {"train": trainloader, "val": valloader}

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
            device=device,
        )
    elif args.data =="china":
        model = FinNLPCL(
            text_embed_dim=768,
            intraday_hiddenDim=256,
            interday_hiddenDim=256,
            intraday_numLayers=1,
            interday_numLayers=1,
            use_attn1=args.use_attn1,
            use_attn2=args.use_attn2,
            maxlen=10,
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
          device=device
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
          device=device
      )
model.to(device)

criterion_xe = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


if os.path.exists("saved_models") == False:
    os.mkdir("saved_models")

loss_history = {"train": [], "val": []}
loss_history_mse = {"train": [], "val": []}
loss_history_xe = {"train": [], "val": []}


accuracy_history = {"train": [], "val": []}
mcc_history = {"train": [], "val": []}


since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 9999999
best_loss_mse = 9999999
best_loss_xe = 9999999
best_epoch = 0
best_mcc = 0
best_acc = 0

for epoch in range(args.num_epochs):
    print("Epoch {}/{}".format(epoch, args.num_epochs - 1))
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["train"]:
        if phase == "train":
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_loss_mse = 0.0
        running_loss_xe = 0.0

        truelabels = []
        predlabels = []
        # Iterate over data.
        for batch_data in dataloaders[phase]:
            embedding_data = batch_data["embedding"]
            embedding_data = embedding_data.to(device)

            truelabels.extend(batch_data["movement_label"].numpy())
            target_price = batch_data["movement_label"]
            target_price = target_price.type(torch.LongTensor).to(device)
        
            target_vol = batch_data["volatility"]
            target_vol[torch.isnan(target_vol)] = 0
            target_vol[torch.isinf(target_vol)] = 0
            target_vol = target_vol.type(torch.FloatTensor).to(
                device).unsqueeze(-1)


            length = batch_data["length_data"]

            if args.model == "time":
                time_feats = batch_data["time_feature"].to(device).squeeze(-1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(phase == "train"):
                if args.model == "simple":
                    outputs_vol, outputs_price = model(embedding_data, length)
                elif args.model == "time":
                    outputs_vol, outputs_price = model(embedding_data, length, time_feats)

                loss_xe = criterion_xe(outputs_price, target_price)
                loss_mse = criterion_mse(outputs_vol, target_vol)

                loss = loss_xe + loss_mse

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * target_price.size(0)
            running_loss_mse += loss_mse.item() * target_price.size(0)
            running_loss_xe += loss_xe.item() * target_price.size(0)

            predlabels.extend(torch.argmax(
                outputs_price, dim=-1).cpu().detach().numpy())
        
        predlabels = np.array(predlabels)
        truelabels = np.array(truelabels)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_loss_mse = running_loss_mse / len(dataloaders[phase].dataset)
        epoch_loss_xe = running_loss_xe / len(dataloaders[phase].dataset)
        loss_history[phase].append(epoch_loss)
        loss_history_mse[phase].append(epoch_loss_mse)
        loss_history_xe[phase].append(epoch_loss_xe)
        epoch_accuracy = accuracy_score(truelabels, predlabels)
        epoch_mcc = matthews_corrcoef(truelabels, predlabels)
        accuracy_history[phase].append(epoch_accuracy)
        mcc_history[phase].append(epoch_mcc)

        
        if epoch_loss < best_loss:
          
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_mcc = epoch_mcc
            best_acc = epoch_accuracy
            best_loss_mse = epoch_loss_mse
            best_loss_xe = epoch_loss_xe

            
        torch.save(
            {
                "model_wts": model.state_dict(),
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "best_loss_mse": best_loss_mse,
                "best_loss_xe": best_loss_xe,
                "best_accuracy": best_acc,
                "best_mcc": best_mcc,
                "loss_history": loss_history,
                "loss_history_mse": loss_history_mse,
                "loss_history_xe": loss_history_xe,
                "mcc_history": mcc_history,
                "accuracy_history": accuracy_history,
                "best_model_wts": best_model_wts,
                "args": args,
            },
            "saved_models/" + 'mtl'+args.data+str(bucket_number) + ".pth",
        )

        print(
            "Epoch: {} Acc: {:.4f}  MCC: {:.4f}  Val BEST Loss: {:.4f}  Val XELoss: {:.4f}  Val MSELoss: {:.4f} ".format(
                epoch, best_acc, best_mcc, best_loss, best_loss_xe, best_loss_mse
            )
        )

        print(
            "{} Epoch: {} Acc: {:.4f} MCC: {:.4f} Loss: {:.4f} XELoss: {:.4f} MSELoss: {:.4f}".format(
                phase, epoch, epoch_accuracy, epoch_mcc, epoch_loss, epoch_loss_xe, epoch_loss_mse
            )
        )

    print()

time_elapsed = time.time() - since
print(
    "Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60)
)
print("Best Train Loss: {:4f}".format(best_loss))

