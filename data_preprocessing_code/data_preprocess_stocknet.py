import pandas as pd
import os
import datetime
import json
import transformers
import torch
import pickle as pkl
from tqdm import tqdm


mean_length = 30
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

    # store the price features
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
        # padded embeddings are zero
        bert_embedding  = torch.zeros(mean_length, 768)
        # padded time feature is 1
        time_diff = torch.ones(mean_length,1)
        tweets = []
        created_at = []

        with open(date_path) as f:
            for line in f:
                data = json.loads(line)
                tweets.append(data['text'])
                created_at.append(datetime.datetime.strptime(data['created_at'], "%a %b %d %H:%M:%S +0000 %Y").replace(second=0, microsecond=0))

        # recent to oldest
        zipped_list = sorted(zip(created_at, tweets), reverse=True)

        # truncate older tweets if above 30
        if(len(zipped_list) > mean_length):
            zipped_list = zipped_list[:mean_length]

        zipped_list.reverse() # oldest to recent
        length_list.append(len(zipped_list))
        created_at, tweets = zip(*zipped_list)

        encoded_inputs = tokenizer(tweets, is_split_into_words=True, padding=True, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(device)
        # has 0 for padded stuff and 1 for normal
        attention_mask = encoded_inputs['attention_mask'].to(device)
        # normalize the attention weights, so that 1->1/Num_tweets and 0->0
        norm_weights =  torch.nn.functional.normalize(attention_mask.float(), p=1, dim=1)
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0].detach()
        # used the normalized weights to calculate average by multiplying and then adding
        weighted_hidden_states = norm_weights.unsqueeze(2)*last_hidden_states
        avg_last = torch.sum(weighted_hidden_states, dim=1)

        # skipped first data point since there is none before it
        for i in range(1,len(created_at)):
            # calc time difference in minutes
            delta_time = created_at[i] - created_at[i-1]
            delta_min = int(delta_time.total_seconds() / 60)
            # if not the same min
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
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
model = transformers.BertModel.from_pretrained('bert-base-cased')
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
        # the input containts i, i+1, i+2, i+3, i+4
        date_window = date_list[i:i+5]
        # the day we want to predict on =>i+5
        target_date = datetime.datetime.strptime(date_list[i+5], '%Y-%m-%d')
        # the last date of the input i+4
        last_date = datetime.datetime.strptime(date_list[i+4], '%Y-%m-%d')
        # labels on the target date
        movement_percent = movement_percent_list[i+5]
        volatility = volatility_list[i+5]

        # if the actual day after the last date is a trading day, it will be equal to the target date, also keep only samples with movement percent > 0.5% or less than -.055%
        if (last_date + datetime.timedelta(days=1) == target_date) and (movement_percent > 0.005 or movement_percent < -0.0055):
            flag = True
            # if the corresponding tweets don't exist, drop
            for date in date_window:
                if not os.path.exists(os.path.join(path_tweet_data, stock, date)):
                    flag = False
                    break
            if flag:
                label = 0
                if movement_percent > 0 :
                    label = 1
                
                date_window_path = [os.path.join(path_tweet_data, stock, x) for x in date_window]
                windowed_data.append({'stock':stock,'dates':date_window,'date_path': date_window_path, 'movement_label': label, 'volatility':volatility})
    
for i in tqdm(range(len(windowed_data))):
    embedding, time_feature, length_data, price_feature = windowed_data_to_embedding(windowed_data[i], model)
    model_data.append({'index':i, 'embedding': embedding,'time_feature': time_feature, 'movement_label':windowed_data[i]['movement_label'], 'volatility':windowed_data[i]['volatility'], "length_data":length_data, 'price_feature':price_feature })
    

with open('data_stocknet.pkl','wb') as f:
    pkl.dump(model_data, f)
