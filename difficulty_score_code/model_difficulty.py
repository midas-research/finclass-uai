# do random seeding
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import FinCLData
import argparse
from model import *
import numpy as np
import random
import pickle as pkl
import time



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
    "--data",
    default="stock",
    help="data to be used [stock, china] (default: stock)",
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
device = torch.device("cuda")


if args.data == "stock":
    traindata = FinCLData("Enter path to train_data_stocknet.pkl here")
elif args.data == "china":
    traindata = FinCLData("Enter path to train_data_chinese.pkl here")

model_difficulty = torch.zeros(len(traindata))
buckets = 10
data_in_each = int(len(traindata) / buckets)
number_in_last = len(traindata) - ((buckets - 1) * data_in_each)
train_buckets = [
    traindata[i * data_in_each : (i + 1) * data_in_each] for i in range(0, buckets - 1)
]
train_buckets.append(
    traindata[
        (buckets - 1) * data_in_each : (buckets - 1) * data_in_each + number_in_last
    ]
)
bucket_number = int(args.bucket)
criterion_xe = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

model_paths = [
    "saved_models/" + "mtl" + args.data + str(x) + ".pth" for x in range(buckets)
]
models = []

for path in model_paths:

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
                device=device,
            )

    model.load_state_dict(torch.load(path)["model_wts"])
    model.to(device)
    model.eval()
    models.append(model)


for i in tqdm(range(len(traindata))):

    bucket_no_data_point = int(i / data_in_each)
    embedding_data = traindata[i]["embedding"]
    embedding_data = embedding_data.to(device)
    length = traindata[i]["length_data"]

    target_price = torch.tensor(traindata[i]["movement_label"])
    target_price = target_price.to(device).unsqueeze(0)

    target_vol = torch.tensor(traindata[i]["volatility"])
    target_vol[torch.isnan(target_vol)] = 0
    target_vol[torch.isinf(target_vol)] = 0
    target_vol = target_vol.type(torch.FloatTensor).to(device).unsqueeze(0)

    if args.model == "time":
        time_feats = traindata[i]["time_feature"].to(device).squeeze(-1)
        time_feats = time_feats.unsqueeze(0)

    embedding_data = embedding_data.unsqueeze(0)
    length = length.unsqueeze(0)

    loss_list = []

    for j in range(buckets):

        if j != bucket_no_data_point:

            model_j = models[j]

            if args.model == "simple":
                outputs_vol, outputs_price = model_j(embedding_data, length)
            elif args.model == "time":
                outputs_vol, outputs_price = model_j(embedding_data, length, time_feats)

            loss_xe = criterion_xe(outputs_price, target_price)
            loss_mse = criterion_mse(outputs_vol, target_vol)
            loss = loss_mse + loss_xe
            loss_list.append(loss)

    loss_vector = torch.stack(loss_list, dim=-1)
    difficulty = torch.mean(loss_vector).detach()
    model_difficulty[i] = difficulty

if args.data == "stock":
    with open("model_difficulty.pkl", "wb") as f:
        pkl.dump(model_difficulty, f)
elif args.data == "china":
    with open("model_difficulty_chinese.pkl", "wb") as f:
        pkl.dump(model_difficulty, f)
