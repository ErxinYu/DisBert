import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,voc_size=2000, n_topic=50, dropout=0.0):

        super(VAE, self).__init__()
        encode_dims=[voc_size,1024,300,n_topic]
        decode_dims=[n_topic,64,voc_size]
        self.encoder = nn.ModuleDict({
            f'enc_{i}':nn.Linear(encode_dims[i],encode_dims[i+1]) 
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2],encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2],encode_dims[-1]) 
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1],encode_dims[-1])

        self.v  = nn.Embedding(decode_dims[-1], decode_dims[-2]) 
        self.t = nn.Embedding(decode_dims[0], decode_dims[-2]) 
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.t.weight)
        self.theta = None
        self.beta = None       
        
    def encode(self, x):
        hid = x
        for i,layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid) 
        return mu, log_var

    def inference(self,x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x,dim=1)
        return theta

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, theta): 
        self.beta = torch.mm(self.t.weight, self.v.weight.transpose(0,1))
        p_x = torch.mm(theta, self.beta)
        return p_x    


    def forward(self, x, use_gsm=True):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 
        if use_gsm:
            self.theta = torch.softmax(_theta,dim=1)
        else:
            self.theta = _theta
        x_reconst = self.decode(self.theta)
        self.beta = torch.mm(self.t.weight, self.v.weight.transpose(0,1))
        return x_reconst, mu, log_var, self.theta, self.beta, self.t.weight

