# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc_h = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_c = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, enc_hid_dim * 2]
        # hidden: [2 * num_layers, batch_size, enc_hid_dim]
        # cell: [2 * num_layers, batch_size, enc_hid_dim]
        
        # Concatenate forward and backward final hidden/cell states
        hidden_last = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        cell_last = self.dropout(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))
        # hidden_last / cell_last: [batch_size, enc_hid_dim * 2]
        
        # Map to Decoder space
        dec_hidden = torch.tanh(self.fc_h(hidden_last)).unsqueeze(0)
        dec_cell = torch.tanh(self.fc_c(cell_last)).unsqueeze(0)
        # dec_hidden / dec_cell: [1, batch_size, dec_hid_dim]
        
        return outputs, (dec_hidden, dec_cell)

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim=128):
        super().__init__()
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: [1, batch_size, dec_hid_dim] -> [batch_size, dec_hid_dim]
        # enc_outputs: [batch_size, src_len, enc_hid_dim * 2]
        batch_size = enc_outputs.shape[0]
        src_len = enc_outputs.shape[1]
        
        # Replicate hidden state src_len times
        hidden = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hid_dim]
        
        # Concat hidden state and encoder outputs to compute energy
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        # energy: [batch_size, src_len, attn_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout=0.4):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # LSTM input is combination of target embedding + context vector
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        
        # Prediction layer takes LSTM output + context vector + input embedding
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, enc_outputs):
        # input: [batch_size] -> unsqueeze to [batch_size, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]
        
        # Compute attention weights
        a = self.attention(hidden, enc_outputs)
        # a: [batch_size, src_len]
        
        a = a.unsqueeze(1)
        # a: [batch_size, 1, src_len]
        
        # Compute weighted context vector
        # context = a * enc_outputs -> [batch_size, 1, enc_hid_dim * 2]
        context = torch.bmm(a, enc_outputs)
        
        # Concatenate embedded token and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [batch_size, 1, emb_dim + enc_hid_dim * 2]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: [batch_size, 1, dec_hid_dim]
        # hidden/cell: [1, batch_size, dec_hid_dim]
        
        # Concatenate all features to classify next token
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        
        prediction = self.fc_out(self.dropout(torch.cat((output, context, embedded), dim=1)))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        
    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        
        if trg is not None:
            trg_len = trg.shape[1]
        else:
            trg_len = 30
            
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Get all encoder outputs for attention calculations
        enc_outputs, (hidden, cell) = self.encoder(src)
        
        # First input token is SOS
        if trg is not None:
            input_token = trg[:, 0]
        else:
            input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
            
        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, enc_outputs)
            outputs[:, t, :] = output
            
            # Select next input token based on teacher forcing ratio
            teacher_force = random.random() < teacher_forcing_ratio
            top_prediction = output.argmax(1)
            
            if trg is not None and t < trg_len - 1:
                input_token = trg[:, t+1] if teacher_force else top_prediction
            else:
                input_token = top_prediction
                
        return outputs
