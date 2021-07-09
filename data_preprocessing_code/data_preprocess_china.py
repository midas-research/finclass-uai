import pandas as pd
import numpy as np
import os
import datetime
import transformers
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

def windowed_data_to_embedding(windowed_data, model):

    date_list = windowed_data['dates']
    date_path_list = windowed_data['date_path']
    stock = windowed_data['stock']
    df = pd.read_csv(os.path.join(path_price_data, stock + '.csv'))

    high = []
    low = []
    close = []

    for date in date_list:
        date_data = df.loc[df['Date'] == date]
        high.append(date_data['High'].values[0])
        low.append(date_data['Low'].values[0])
        close.append(date_data['Close'].values[0])

    high = torch.tensor(high).unsqueeze(1)
    low = torch.tensor(low).unsqueeze(1)
    close = torch.tensor(close).unsqueeze(1)

    price_feature = torch.cat((high, low, close), dim=1)

    embedding_list = []
    time_feature_list = []
    length_list = []

    for date_path in date_path_list:
        bert_embedding  = torch.zeros(mean_length, 768)
        time_diff = torch.ones(mean_length,1)
        tweets = []
        created_at = []

        df = pd.read_csv(date_path)
        tweets = df['text'].values
        time_created = df['Time'].values

        for time in time_created:
            created_at.append(datetime.datetime.strptime(time, "%H:%M:%S"))

        zipped_list = sorted(zip(created_at, tweets), reverse=True)
        if(len(zipped_list) > mean_length):
            zipped_list = zipped_list[:mean_length]

        zipped_list.reverse() 
        length_list.append(len(zipped_list))
        created_at, tweets = zip(*zipped_list)

        encoded_inputs = tokenizer(tweets, padding=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        norm_weights =  torch.nn.functional.normalize(attention_mask.float(), p=1, dim=1)
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0].detach()
        weighted_hidden_states = norm_weights.unsqueeze(2)*last_hidden_states
        avg_last = torch.sum(weighted_hidden_states, dim=1)

        for i in range(1,len(created_at)):
            delta_time = created_at[i] - created_at[i-1]
            delta_min = int(delta_time.total_seconds() / 60)
            if(delta_min!=0):
                time_diff[i] = 1/delta_min
        
        bert_embedding[:avg_last.shape[0]] = avg_last

        embedding_list.append(bert_embedding)
        time_feature_list.append(time_diff)

    embedding_windowed = torch.stack(embedding_list, dim=0)
    time_feature_windowed = torch.stack(time_feature_list, dim=0)
    length_windowed = torch.tensor(length_list)

    return embedding_windowed, time_feature_windowed, length_windowed, price_feature


cwd = os.getcwd()
path_tweet_data  = "Enter path to tweet data here"
path_price_data = "Enter path to price data here"
stock_names = os.listdir(path_tweet_data)
windowed_data = {}
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')
model = transformers.BertModel.from_pretrained('bert-base-chinese')
model = model.to(device)


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
                
                date_window_path = [os.path.join(path_tweet_data, stock, x+'.csv') for x in date_window]
                windowed_data.append({'stock':stock,'dates':date_window,'date_path': date_window_path, 'movement_label': label, 'volatility':volatility})

for i in tqdm(range(len(windowed_data))):
    embedding, time_feature, length_data, price_feature = windowed_data_to_embedding(windowed_data[i], model)
    model_data.append({'index':i, 'embedding': embedding,'time_feature': time_feature, 'movement_label':windowed_data[i]['movement_label'], 'volatility':windowed_data[i]['volatility'], "length_data":length_data, 'price_feature':price_feature })
    

with open('data_chinese.pkl','wb') as f:
    pkl.dump(model_data, f)
