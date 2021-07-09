import pandas as pd
import os
import datetime
import torch
import pickle as pkl
from tqdm import tqdm


mean_length = 10 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def check_consecutive(dates):
    for i in range(len(dates)-1):
        if((dates[i] + datetime.timedelta(days=1))!=dates[i+1]):
            return False
        
    return True

cwd = os.getcwd()
path_tweet_data  = "Enter path to tweet data here"
path_price_data = "Enter path to price data here"
stock_names = os.listdir(path_tweet_data)
windowed_data = {}

stock_to_tweet_embeddings = {}
windowed_data = []
model_data = []

for (i,stock) in enumerate(stock_names):
    path_stock = os.path.join(path_price_data, stock+'.csv')
    df = pd.read_csv(path_stock)
    date_list = df['Date'].values[::-1]
    movement_percent_list = df['Movement Percent'].values[::-1]
    volatility_list = df['Volatility'].values[::-1]

    for i in range(len(date_list) - 5):
        date_window = date_list[i:i+5]
        target_date = datetime.datetime.strptime(date_list[i+5], '%Y-%m-%d')
        last_date = datetime.datetime.strptime(date_list[i+4], '%Y-%m-%d')
        movement_percent = movement_percent_list[i+5]
        volatility = volatility_list[i+5]

        if (last_date + datetime.timedelta(days=1) == target_date) :
            flag = True
            for date in date_window:
                if not os.path.exists(os.path.join(path_tweet_data, stock, date+ '.csv')):
                    flag = False
                    break
            if flag:
                label = 0
                if movement_percent > 0 :
                    label = 1
                
                windowed_data.append({'date_list': date_window, 'movement_label': label, 'volatility':volatility})
    
train_split = []
val_split = []
test_split =  []

for i in tqdm(range(len(windowed_data))):
    date_data = windowed_data[i]['date_list']
    last_date = datetime.datetime.strptime(date_data[-1], '%Y-%m-%d')
    first_date = datetime.datetime.strptime(date_data[0], '%Y-%m-%d')
    target_date = last_date + datetime.timedelta(days=1)

    if target_date.month <= 9:
        train_split.append(i)

    elif first_date.month >=10:
        test_split.append(i)

with open('data_chinese.pkl','rb') as f:
    data = pkl.load(f)

train_data = [data[x] for x in train_split]
test_data = [data[x] for x in test_split]

with open('train_split_chinese.pkl','wb') as f:
    pkl.dump(train_split, f)

with open('test_split_chinese.pkl','wb') as f:
    pkl.dump(test_split, f)

with open('train_data_chinese.pkl','wb') as f:
    pkl.dump(train_data, f)

with open('test_data_chinese.pkl','wb') as f:
    pkl.dump(test_data, f)