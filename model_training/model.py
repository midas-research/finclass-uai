import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, hidden_states, reverse=False):

        b, seq, embed = inputs.size()
        h = hidden_states[0]
        c = hidden_states[1]

        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        hidden_state_h = []
        hidden_state_c = []

        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))  # short term mem
            # discounted short term mem
            c_s2 = c_s1 * timestamps[:, s : s + 1].expand_as(c_s1)
            c_l = c - c_s1  # long term mem
            c_adj = c_l + c_s2  # adjusted = long + disc short term mem
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(o)
            hidden_state_c.append(c)
            hidden_state_h.append(h)

        if reverse:
            outputs.reverse()
            hidden_state_c.reverse()
            hidden_state_h.reverse()

        outputs = torch.stack(outputs, 1)
        hidden_state_c = torch.stack(hidden_state_c, 1)
        hidden_state_h = torch.stack(hidden_state_h, 1)

        return outputs, (h, c)


class attn(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(attn, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape)
            self.W2 = torch.nn.Linear(in_shape, in_shape)
            self.V = torch.nn.Linear(in_shape, 1)
        if maxlen != None:
            self.arange = torch.arange(maxlen)

    def forward(self, full, last, lens=None, dim=1):
        """
        full : B*30*in_shape
        last : B*1*in_shape
        lens: B*1
        """
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))

            if lens != None:
                mask = self.arange[None, :] < lens[:, None]
                score[~mask] = float("-inf")

            attention_weights = F.softmax(score, dim=dim)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        else:
            if lens != None:
                mask = self.arange[None, :] < lens[:, None]
                mask = mask.type(torch.float).unsqueeze(-1).cuda()
                context_vector = full * mask
                context_vector = torch.mean(context_vector, dim=dim)
                return context_vector
            else:
                return torch.mean(full, dim=dim)


class FinNLPCL(nn.Module):
    def __init__(
        self,
        text_embed_dim,
        intraday_hiddenDim,
        interday_hiddenDim,
        intraday_numLayers,
        interday_numLayers,
        use_attn1,
        use_attn2,
        maxlen=30,
        outdim=2,
        device=torch.device("cpu"),
    ):
        super(FinNLPCL, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=intraday_hiddenDim,
            num_layers=intraday_numLayers,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm1_outshape = intraday_hiddenDim
        self.attn1 = attn(
            in_shape=self.lstm1_outshape, use_attention=use_attn1, maxlen=maxlen
        )
        self.maxlen = maxlen
        self.lstm2 = nn.LSTM(
            input_size=self.lstm1_outshape,
            hidden_size=interday_hiddenDim,
            num_layers=interday_numLayers,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2_outshape = interday_hiddenDim
        self.attn2 = attn(in_shape=self.lstm2_outshape, use_attention=use_attn2)

        self.linear3 = nn.Linear(self.lstm2_outshape, 256)
        self.linear4 = nn.Linear(256, outdim)

        self.drop = nn.Dropout(p=0.3)
        self.batchnorm = nn.BatchNorm1d(128)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device = device

    def forward(self, sentence_feats, len_tweets):
        """
        sentence_feat: sentence features (B*5*30*N),
        len_tweets: (B*5)
        """
        sentence_feats = sentence_feats.permute(1, 0, 2, 3)
        len_tweets = len_tweets.permute(1, 0)
        len_days = sentence_feats.size(0)
        bs = sentence_feats.size(1)
        lstm1_out = torch.zeros(len_days, bs, self.lstm1_outshape).to(self.device)

        for i in range(len_days):
            temp = pack_padded_sequence(
                sentence_feats[i], len_tweets[i], batch_first=True, enforce_sorted=False
            )
            temp_lstmout, (temp_hn, _) = self.lstm1(temp)
            temp_lstmout, _ = pad_packed_sequence(temp_lstmout, batch_first=True)
            temp1 = torch.zeros(bs, self.maxlen, self.lstm1_outshape).to(self.device)
            batchmaxlen = temp_lstmout.size(1)
            temp1[:, :batchmaxlen, :] = temp_lstmout
            temp_hn = temp_hn.permute(1, 0, 2)
            lstm1_out[i] = self.attn1(temp1, temp_hn, len_tweets[i])

        lstm1_out = lstm1_out.permute(1, 0, 2)
        lstm2_out, (hn2_out, _) = self.lstm2(lstm1_out)
        hn2_out = hn2_out.permute(1, 0, 2)
        x = self.attn2(lstm2_out, hn2_out)
        x = self.drop(self.relu(self.linear3(x)))
        x = self.linear4(x)

        return x


class TimeFinNLPCL(nn.Module):
    """
    Forecasting model using LSTM

    B*5*30*N
    """

    def __init__(
        self,
        text_embed_dim,
        intraday_hiddenDim,
        interday_hiddenDim,
        intraday_numLayers,
        interday_numLayers,
        use_attn1,
        use_attn2,
        maxlen=30,
        outdim=2,
        device=torch.device("cpu"),
    ):
        super(TimeFinNLPCL, self).__init__()
        self.lstm1 = TimeLSTM(
            input_size=text_embed_dim,
            hidden_size=intraday_hiddenDim,
        )
        self.intraday_hiddenDim = intraday_hiddenDim
        self.lstm1_outshape = intraday_hiddenDim
        self.attn1 = attn(
            in_shape=self.lstm1_outshape, use_attention=use_attn1, maxlen=maxlen
        )
        self.maxlen = maxlen
        self.lstm2 = nn.LSTM(
            input_size=self.lstm1_outshape,
            hidden_size=interday_hiddenDim,
            num_layers=interday_numLayers,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm2_outshape = interday_hiddenDim
        self.attn2 = attn(self.lstm2_outshape, use_attention=use_attn2)

        self.linear3 = nn.Linear(self.lstm2_outshape, 256)
        self.linear4 = nn.Linear(256, outdim)

        self.drop = nn.Dropout(p=0.3)
        self.batchnorm = nn.BatchNorm1d(128)
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.device = device

    def init_hidden(self):
        h = Variable(torch.zeros(self.bs, self.intraday_hiddenDim)).to(self.device)
        c = Variable(torch.zeros(self.bs, self.intraday_hiddenDim)).to(self.device)

        return (h, c)

    def forward(self, sentence_feats, len_tweets, time_feats):
        """
        sentence_feat: sentence features (B*5*30*N),
        len_tweets: (B*5)
        time_feats: (B*5*30)
        """
        sentence_feats = sentence_feats.permute(1, 0, 2, 3)
        len_days, self.bs, _, _ = sentence_feats.size()
        h_init, c_init = self.init_hidden()

        len_tweets = len_tweets.permute(1, 0)
        time_feats = time_feats.permute(1, 0, 2)

        lstm1_out = torch.zeros(len_days, self.bs, self.lstm1_outshape).to(self.device)

        for i in range(len_days):
            temp_lstmout, (_, _) = self.lstm1(
                sentence_feats[i], time_feats[i], (h_init, c_init)
            )
            last_idx = len_tweets[i]
            last_idx = last_idx.type(torch.int).tolist()
            temp_hn = torch.zeros(self.bs, 1, self.lstm1_outshape)
            for j in range(self.bs):
                temp_hn = temp_lstmout[j, last_idx[j] - 1, :]
            lstm1_out[i] = self.attn1(temp_lstmout, temp_hn, len_tweets[i])

        lstm1_out = lstm1_out.permute(1, 0, 2)
        lstm2_out, (hn2_out, _) = self.lstm2(lstm1_out)
        hn2_out = hn2_out.permute(1, 0, 2)
        x = self.attn2(lstm2_out, hn2_out)
        x = self.drop(self.relu(self.linear3(x)))
        x = self.linear4(x)

        return x
