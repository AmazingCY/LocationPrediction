import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum
import math


class Rnn(Enum):
    ''' The available RNN units '''

    RNN = 0
    GRU = 1
    LSTM = 2
    Transformer = 3

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        if name == 'transformer':
            return Rnn.Transformer
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    ''' Creates the desired RNN unit. '''

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'
        if self.rnn_type == Rnn.Transformer:
            return "Use pytorch Transformer implementation."

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def is_transformer(self):
        return self.rnn_type in [Rnn.Transformer]

    def create(self, hidden_size, batch_first_sign=False):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size, batch_first=batch_first_sign)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size, bidirectional=True, bias=True, num_layers=1)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size, bidirectional=True, bias=True, num_layers=1)
        if self.rnn_type == Rnn.Transformer:
            return nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)


class LocationPrediction_RNN(nn.Module):
    ''' Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    '''

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        # RNN
        self.input_size = input_size  # loc number
        self.user_count = user_count  # user number
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc_bi = nn.Linear(3 * hidden_size, input_size)  # create outputs in lenght of locations
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.dropout = nn.Dropout(0.5)
        # Attention parameter
        '''
        self.w_q = nn.Parameter(torch.randn(10, 2 * hidden_size, hidden_size))
        self.w_k = nn.Parameter(torch.randn(10, 2 * hidden_size, hidden_size))
        self.w_v = nn.Parameter(torch.randn(10, 2 * hidden_size, hidden_size))
        '''
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        # Transformer
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.rnn, num_layers=3)

    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score  # [batch, seq_len, hidden_dim*2]
        # context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim*2]
        return scored_x

    def self_attention_net(self, x, query, mask=None):
        d_k = query.size(-1)  # d_k为query的维度
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
        #         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        #  scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(self.hidden_size * 2)
        #         print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
        #  print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x)

        return context

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        '''
        :param x: [seq_length,batch]
        :param t: [seq_length,batch]
        :param s:  [seq_length,batch,2]
        :param y_t:  [seq_length,batch]
        :param y_s: [seq_length,batch,2]
        :param h: [layer,batch,hidden_size]
        :param active_user:[1,batch]
        :return:  y_liner,h
        '''
        seq_len, user_len = x.size()
        # print(x)
        x_emb = self.encoder(x)
        out, h = self.rnn(x_emb, h)  # [seq_length,batch,hidden_size]
        # comopute weights per user

        out_w_forward = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        out_w_backward = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)

        for i in range(seq_len):
            sum_w_forward = torch.zeros(user_len, 1, device=x.device)
            sum_w_backward = torch.zeros(user_len, 1, device=x.device)
            out_forward, out_backward = torch.split(out, self.hidden_size, dim=2)
            for j in range(i + 1):
                dist_t_forward = t[i] - t[j]
                dist_t_backward = t[seq_len - 1 - j] - t[seq_len - 1 - i]
                dist_s_forward = torch.norm(s[i] - s[j], dim=-1)
                dist_s_backward = torch.norm(s[seq_len - 1 - j] - s[seq_len - 1 - i], dim=-1)
                a_j_forward = self.f_t(dist_t_forward, user_len)
                b_j_forward = self.f_s(dist_s_forward, user_len)
                a_j_forward = a_j_forward.unsqueeze(1)
                b_j_forward = b_j_forward.unsqueeze(1)
                a_j_backward = self.f_t(dist_t_backward, user_len)
                b_j_backward = self.f_s(dist_s_backward, user_len)
                a_j_backward = a_j_backward.unsqueeze(1)
                b_j_backward = b_j_backward.unsqueeze(1)
                w_j_forward = a_j_forward * b_j_forward + 1e-10  # small epsilon to avoid 0 division
                w_j_backward = a_j_backward * b_j_backward + 1e-10
                sum_w_forward += w_j_forward
                sum_w_backward += w_j_backward
                out_w_forward[i] += w_j_forward * out_forward[j]
                out_w_backward[seq_len - 1 - i] += w_j_backward * out_w_backward[
                    seq_len - 1 - j]
            # normliaze according to weights
            out_w_forward[i] /= sum_w_forward
            out_w_backward[seq_len - 1 - i] /= sum_w_backward
            # print(out_w.size())

        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)  # (batch,hidden_size)
        # self_attention layer
        out_w = torch.cat([out_w_forward, out_w_backward], 2)  # [seq_length,batch,hidden_size*2]
        out_w = out_w.permute(1, 0, 2)  # [batch,seq_length,hidden_size*2]
        query = self.dropout(out_w)
        attn_output = self.self_attention_net(out_w, query)
        attn_output = attn_output.permute(1, 0, 2)
        # print(p_u.size())
        out_pu = torch.zeros(seq_len, user_len, 3 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([attn_output[i], p_u], dim=1)

        y_linear = self.fc_bi(out_pu)
        return y_linear, h


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(2, self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(2, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h, c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h, c)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LocationPrediction_Transformer(nn.Module):
    def __init__(self, input_size, user_count, hidden_size, rnn_factory):
        super().__init__()

        self.model_type = 'Transformer'
        self.input_size = input_size  # loc number
        self.user_count = user_count  # user number
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, hidden_size)  # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc_bi = nn.Linear(3 * hidden_size, input_size)  # create outputs in lenght of locations
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.dropout = nn.Dropout(0.5)

        self.src_mask = None
        self.norm = LayerNorm(hidden_size)
        #self.pos_encoder = WindowRNN(hidden_size, 12)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.normalize_layer = NormalizeLayer(hidden_size, 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.rnn, num_layers=3) # seq_length * batch * dim

    def generate_square_subsequent_mask(self, src, seq_length, user_length):
        '''
        padding_mask
        src:max_lenth,num,300
        lenths:[lenth1,lenth2...]
        '''
        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(0), src.size(1)) == 1
        for i in range(seq_length):
            for j in range(user_length):
                mask[i][j] = False
        return mask

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        seq_len, user_len = x.size()  # seq_length * batch_size
        x_emb = self.encoder(x)  # seq_length * batch_size * hidden_size
        #x_emb_norm = self.norm(x_emb.permute(1, 0, 2))
        x_emb_pos_encoder = self.pos_encoder(x_emb)  # batch_size * seq_length * dim
        #x_emb_pos_norm = self.normalize_layer(x_emb, x_emb_pos_encoder)  # batch_size * seq_length * dim
        # src_mask = self.generate_square_subsequent_mask(x_emb, seq_len, user_len)
        out = self.transformer_encoder(x_emb_pos_encoder)  # seq_length * batch_size * hidden_size
        #out_norm = self.normalize_layer(x_emb_pos_norm,out)
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out[i], p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear, h


class WindowRNN(nn.Module):
    def __init__(self, hidden_size, window_size):
        super(WindowRNN, self).__init__()
        """
        LocalRNN structure
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        """
        self.window = window_size
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # input_size * hidden_size

        # index of each window
        idx = [i for j in range(self.window - 1, 10000, 1) for i in range(j - (self.window - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.window - 1, hidden_size)).cuda()

    def forward(self, x):
        '''
        :param x: seq_length * batch_size * hidden_size
        :return h_pos_encoder: batch_size * seq_length * dim
        '''
        input_seq = x  # batch_size * seq_length * hidden_size
        input_window = self.get_K(input_seq)  # batch_size * seq_length * window_size * dim
        batch, length, window_size, d_model = input_window.shape
        input_rnn = input_window.view(-1, window_size, d_model)  # batch_size x seq_length * window_size * dim
        h_pos_encoder = self.rnn(input_rnn)[0][:, -1, :]  # batch_size * dim
        return h_pos_encoder.view(batch, length, d_model)  # batch_size * seq_length * dim

    def get_K(self, x):
        '''
        :param x: batch_size * seq_length * hidden_size
        :return key: batch_size * seq_length * window_size * dim
        '''
        batch_size, seq_length, dim = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.window * seq_length])
        key = key.reshape(batch_size, seq_length, self.window, -1)
        return key


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class NormalizeLayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(NormalizeLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, pre_input, new_output):
        "Apply residual connection to any sublayer with the same size."
        return pre_input.permute(1, 0, 2) + self.dropout(new_output)
