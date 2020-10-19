import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self, vocab_size, output_dim, embedding_dim,
                n_filters, filter_sizes, dropout,
                first_vec, pad_idx, unk_idx):                                
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)      
    self.convs = nn.ModuleList([
                                nn.Conv2d(in_channels = 1, 
                                        out_channels = n_filters, 
                                        kernel_size = (fs, embedding_dim)) 
                                for fs in filter_sizes
                                ])    
    self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)    
    self.dropout = nn.Dropout(dropout)

    # initial weights
    self.embedding.weight.data.copy_(first_vec)
    self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
    self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)      
      
  def forward(self, text):        
    #text: [sent len, batch size] --> [batch size, sent len]            
    text = text.permute(1, 0)

    #embedded: [batch size, sent len, emb dim]
    embedded = self.embedding(text)
            
    #embedded: [batch size, 1, sent len, emb dim]
    embedded = embedded.unsqueeze(1)
    
    #conv_n: [batch size, n_filters, sent len - filter_sizes[n]]
    conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
    #pooled_n: [batch size, n_filters]; len(pooled) == len(filter_sizes)
    pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

    #cat: [batch size, n_filters * len(filter_sizes)]
    cat = self.dropout(torch.cat(pooled, dim = 1))
    return self.fc(cat)


class GRU(nn.Module):
  def __init__(self, vocab_size, output_dim, embedding_dim, 
              hidden_dim, n_layers, bidirectional,
              dropout, first_vec, pad_idx, unk_idx): 
    super().__init__()                      
    self.embedding = nn.Embedding(vocab_size, embedding_dim)      
    self.rnn = nn.GRU(embedding_dim,
                      hidden_dim,
                      num_layers = n_layers,
                      bidirectional = bidirectional,
                      batch_first = True,
                      dropout = 0 if n_layers < 2 else dropout)      
    self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)      
    self.dropout = nn.Dropout(dropout)

    # initial weights
    self.embedding.weight.data.copy_(first_vec)
    self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
    self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)         
      
  def forward(self, text):
      
    #text: [sent len, batch size] --> text: [batch size, sent len]            
    text = text.permute(1, 0)
    
    #embedded: [batch size, sent len, emb dim]
    embedded = self.embedding(text)
            
    #embedded: [batch size, sent len, emb dim]
    #--> 
    #embedded: [batch size, 1, sent len, emb dim]
    #embedded = embedded.unsqueeze(1)
      
    #hidden = [n layers * n directions, batch size, emb dim]
    _, hidden = self.rnn(embedded)
      
      #hidden = [n layers * n directions, batch size, emb dim]
    if self.rnn.bidirectional:
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    else:
        hidden = self.dropout(hidden[-1,:,:])

    #hidden = [batch size, hid dim]

    output = self.out(hidden)

    #output = [batch size, out dim]

    return output

from transformers import BertModel
class BERTGRU(nn.Module):
  def __init__(self, bert, output_dim, hidden_dim, n_layers, bidirectional,
              dropout, frozen = False):      
    super().__init__()
    self.bert = bert      
    embedding_dim = bert.config.to_dict()['hidden_size']
    self.rnn = nn.GRU(embedding_dim,
                      hidden_dim,
                      num_layers = n_layers,
                      bidirectional = bidirectional,
                      batch_first = True,
                      dropout = 0 if n_layers < 2 else dropout)
    self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
    #freezing bert will reduce many millions of parameters to learn
    if frozen:  
      for name, param in self.named_parameters(): 
          if name.startswith('bert'): param.requires_grad = False      
      
  def forward(self, text):                 
    with torch.no_grad():
        embedded = self.bert(text)[0]
            
    #embedded = [batch size, sent len, emb dim]   

    _, hidden = self.rnn(embedded)
    
    #hidden = [n layers * n directions, batch size, emb dim]
    
    if self.rnn.bidirectional:
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    else:
        hidden = self.dropout(hidden[-1,:,:])
            
    #hidden = [batch size, hid dim]
    
    output = self.out(hidden)
    
    #output = [batch size, out dim]
    
    return output


from transformers import BertModel
class BERT_only(nn.Module):
  def __init__(self, bert, output_dim, dropout, pooled=False, frozen=False):      
    super().__init__()      
    self.bert = bert      
    embedding_dim = bert.config.to_dict()['hidden_size']      
    self.pooled = pooled
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(embedding_dim, output_dim)
    #freezing bert will reduce many millions of parameters to learn     
    if frozen:  
      for name, param in self.named_parameters(): 
          if name.startswith('bert'): param.requires_grad = False    

  def forward(self, text):                 
    # aggregating all wordvectors per sentences, the in-built "pooled" != average
    embedded = self.bert(text)[1] if self.pooled else torch.mean(self.bert(text)[0], 1)
    embedded = self.dropout(embedded)  
    output = self.out(embedded)      
    return output


class BERT_only2(nn.Module):
  def __init__(self, bert, output_dim, dropout, frozen=False):      
    super().__init__()      
    self.bert = bert      
    embedding_dim = bert.config.to_dict()['hidden_size']      
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(embedding_dim, output_dim)
    #freezing bert will reduce many millions of parameters to learn
    if frozen:  
      for name, param in self.named_parameters(): 
          if name.startswith('bert'): param.requires_grad = False      

  def forward(self, text):                 
    embedded = self.bert(text)[1]   
    #embedded = torch.mean(embedded, 1) # aggregating all wordvectors per sentences
    embedded = self.dropout(embedded)  
    output = self.out(embedded)      
    return output      
  