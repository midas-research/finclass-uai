import pandas as pd
import numpy as np
import os
import datetime
import json
import transformers
import torch
import pickle as pkl
from tqdm import tqdm
import numpy as np
import itertools
import numpy as np
from pyentrp import entropy as ent
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from pyentrp import entropy as ent


def mpe(mts, m, d):
    # initialize parameters
    n = len(mts[0])
    e = len(mts)
    permutations = np.array(list(itertools.permutations(range(m))))
    t = n - d * (m - 1)
    c = []
    p = []
    pe_channel = []

    for j in range(e):
        c.append([0] * len(permutations))

    # compute single series permutation entropy based on the multivariate distribution of motifs
    for f in range(e):
        for i in range(t):
            sorted_index_array = np.array(
                np.argsort(mts[f][i : i + d * m : d], kind="quicksort")
            )
            for j in range(len(permutations)):
                if abs(permutations[j] - sorted_index_array).any() == 0:
                    c[f][j] += 1

        p.append(np.divide(np.array(c[f]), float(t * e)))
        pe_channel.append(-np.nansum(p[f] * np.log2(p[f])))

    # compute the cross-series permutation entropy based on the multivariate distribution of motifs
    rp = []
    pe_cross = []
    for w in range(len(permutations)):
        rp.append(np.nansum(np.array(p)[:, w]))

    pe_cross = -np.nansum(rp * np.log2(rp))

    return pe_channel, pe_cross


mean_length = 30  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_consecutive(dates):
    for i in range(len(dates) - 1):
        if (dates[i] + datetime.timedelta(days=1)) != dates[i + 1]:
            return False

    return True


def windowed_data_to_difficulty_score(windowed_data):

    date_list = windowed_data["dates"]
    date_path_list = windowed_data["date_path"]
    stock = windowed_data["stock"]
    df = pd.read_csv(os.path.join(path_price_data, stock + ".csv"))

    price_feature = []

    # store the price features
    for date in date_list:
        date_data = df.loc[df["Date"] == date]
        price_feature.append(
            [
                date_data["Open"].values[0],
                date_data["High"].values[0],
                date_data["Low"].values[0],
                date_data["Close"].values[0],
            ]
        )

    pe_cross_list = []
    logits_list = []

    for date_path in date_path_list:

        tweets = []
        created_at = []

        with open(date_path) as f:
            for line in f:
                data = json.loads(line)
                tweets.append(data["text"])
                created_at.append(
                    datetime.datetime.strptime(
                        data["created_at"], "%a %b %d %H:%M:%S +0000 %Y"
                    ).replace(second=0, microsecond=0)
                )

        zipped_list = sorted(zip(created_at, tweets), reverse=True)
        if len(zipped_list) > mean_length:
            zipped_list = zipped_list[:mean_length]

        created_at, tweets = zip(*zipped_list)

        encoded_inputs = tokenizer(
            tweets, is_split_into_words=True, padding=True, return_tensors="pt"
        )
        input_ids = encoded_inputs["input_ids"].to(device)
        # has 0 for padded stuff and 1 for normal
        attention_mask = encoded_inputs["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits1 = logits.permute(1, 0).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        pe_channel, pe_cross = mpe(logits1, 3, 1)
        pe_cross_list.append(pe_cross)
        logits_list.append(logits)

    dtw_list = []

    for i in range(4):
        distance, path = fastdtw(logits_list[i], logits_list[i + 1], dist=euclidean)
        dtw_list.append(distance)

    pe_pe_cross = ent.permutation_entropy(pe_cross_list)
    pe_dtw = ent.permutation_entropy(dtw_list)

    price_feature = np.array(price_feature)
    price_feature = price_feature.transpose()

    _, pe_cross_price_feat = mpe(price_feature, 3, 1)

    entropy = pe_pe_cross + pe_dtw + pe_cross_price_feat

    return entropy, pe_pe_cross + pe_dtw, pe_cross_price_feat


cwd = os.getcwd()
path_tweet_data = "Enter path to tweet data"
path_price_data = "Enter path to price data"
stock_names = os.listdir(path_tweet_data)
windowed_data = {}
tokenizer = transformers.BertTokenizer.from_pretrained("ProsusAI/finbert")
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert"
)
model = model.to(device)
stock_to_tweet_embeddings = {}
windowed_data = []
model_data = []

for (i, stock) in enumerate(stock_names):
    path_stock = os.path.join(path_price_data, stock + ".csv")
    df = pd.read_csv(path_stock)
    date_list = df["Date"].values[::-1]
    movement_percent_list = df["Movement Percent"].values[::-1]
    volatility_list = df["Volatility"].values[::-1]

    for i in range(len(date_list) - 5):
        date_window = date_list[i : i + 5]
        target_date = datetime.datetime.strptime(date_list[i + 5], "%Y-%m-%d")
        last_date = datetime.datetime.strptime(date_list[i + 4], "%Y-%m-%d")
        movement_percent = movement_percent_list[i + 5]
        volatility = volatility_list[i + 5]

        if (last_date + datetime.timedelta(days=1) == target_date) and (
            movement_percent > 0.005 or movement_percent < -0.0055
        ):
            flag = True
            for date in date_window:
                if not os.path.exists(os.path.join(path_tweet_data, stock, date)):
                    flag = False
                    break
            if flag:
                label = 0
                if movement_percent > 0:
                    label = 1

                date_window_path = [
                    os.path.join(path_tweet_data, stock, x) for x in date_window
                ]
                windowed_data.append(
                    {
                        "stock": stock,
                        "dates": date_window,
                        "date_path": date_window_path,
                        "movement_label": label,
                        "volatility": volatility,
                    }
                )

text_price_difficulty_list = []
text_difficulty_list = []
price_difficulty_list = []

for i in tqdm(range(len(windowed_data))):
    text_price_difficulty, text_difficulty, price_difficulty = windowed_data_to_difficulty_score(windowed_data[i])
    text_price_difficulty_list.append(text_price_difficulty)
    text_difficulty_list.append(text_difficulty)
    price_difficulty_list.append(price_difficulty)


with open("Enter path to train_split.pkl here", "rb") as f:
    train_split = pkl.load(f)

text_price_difficulty_train = [text_price_difficulty_list[x] for x in train_split]
text_difficulty_train = [text_difficulty_list[x] for x in train_split]
price_difficulty_train = [price_difficulty_list[x] for x in train_split]

with open("text_price_difficulty.pkl", "wb") as f:
    pkl.dump(text_price_difficulty_train, f)

with open("text_difficulty.pkl", "wb") as f:
    pkl.dump(text_difficulty_train, f)

with open("price_difficulty.pkl", "wb") as f:
    pkl.dump(price_difficulty_train, f)