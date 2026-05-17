# models/seq2seq.py
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.3):
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
        # hidden: [2 * num_layers (2 here), batch_size, enc_hid_dim]
        # cell: [2 * num_layers (2 here), batch_size, enc_hid_dim]
        
        # Concatenate forward (index -2) and backward (index -1) hidden states of the Bi-LSTM
        hidden_last = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        cell_last = self.dropout(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1))
        # hidden_last / cell_last: [batch_size, enc_hid_dim * 2]
        
        # Project states to decoder dimension
        dec_hidden = torch.tanh(self.fc_h(hidden_last)).unsqueeze(0)
        dec_cell = torch.tanh(self.fc_c(cell_last)).unsqueeze(0)
        # dec_hidden / dec_cell: [1, batch_size, dec_hid_dim]
        
        return outputs, (dec_hidden, dec_cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch_size] -> unsqueeze to shape [batch_size, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [batch_size, 1, dec_hid_dim]
        # hidden/cell: [1, batch_size, dec_hid_dim]
        
        # Apply dropout to the hidden output before projecting to vocabulary logits!
        prediction = self.fc_out(self.dropout(output.squeeze(1)))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
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
            trg_len = 30 # Default max word length for inference
            
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        
        # Encode source input sequence
        _, (hidden, cell) = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        if trg is not None:
            input_token = trg[:, 0]
        else:
            input_token = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=self.device)
            
        for t in range(0, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = output
            
            # Determine if teacher forcing is used
            teacher_force = random.random() < teacher_forcing_ratio
            top_prediction = output.argmax(1)
            
            if trg is not None and t < trg_len - 1:
                input_token = trg[:, t+1] if teacher_force else top_prediction
            else:
                input_token = top_prediction
                
        return outputs
