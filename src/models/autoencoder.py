import torch
import torch.nn as nn
from torch.nn import init


def init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, val=0)


class MultichannelAutoEncoder(nn.Module):
    def __init__(self, word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim):
        super(MultichannelAutoEncoder, self).__init__()
        self.wdim = word_dim
        self.wdim1 = word_dim1
        self.wdim2 = word_dim2
        self.idim = image_dim
        self.idim1 = image_dim1
        self.idim2 = image_dim2
        self.zdim = multi_dim

        self.word_encoder = nn.Sequential(
            nn.Linear(self.wdim, self.wdim1),
            nn.Tanh(),
            nn.Linear(self.wdim1, self.wdim2),
            nn.Tanh()
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(self.idim, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim2),
            nn.Tanh()
        )

        self.multi_encoder = nn.Sequential(
            nn.Linear(self.wdim2 + self.idim2, self.zdim),
            nn.Tanh()
        )

        self.multi_decoder = nn.Sequential(
            nn.Linear(self.zdim, self.wdim2 + self.idim2),
            nn.Tanh()
        )

        self.word_decoder = nn.Sequential(
            nn.Linear(self.wdim2, self.wdim1),
            nn.Tanh(),
            nn.Linear(self.wdim1, self.wdim),
            nn.Tanh()
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(self.idim2, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim),
        )

        self.init_parameters()

    def init_parameters(self):
        for net in [self.word_encoder, self.word_decoder,
                    self.image_encoder, self.image_decoder,
                    self.multi_encoder, self.multi_decoder]:
            net.apply(init_weights)

    def forward(self, x_w, x_i):
        encoded_word = self.word_encoder(x_w)
        encoded_image = self.image_encoder(x_i)
        encoded = self.multi_encoder(torch.cat((encoded_word, encoded_image), dim=1))
        decoded = self.multi_decoder(encoded)
        decoded_word = self.word_decoder(decoded[:, :self.wdim2])
        decoded_image = self.image_decoder(decoded[:, -self.idim2:])
        return decoded_word, decoded_image, encoded


class GatedMultichannelAutoEncoder(nn.Module):
    def __init__(self, word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim):
        super(GatedMultichannelAutoEncoder, self).__init__()
        self.wdim = word_dim
        self.wdim1 = word_dim1
        self.wdim2 = word_dim2
        self.idim = image_dim
        self.idim1 = image_dim1
        self.idim2 = image_dim2
        self.zdim = multi_dim

        self.word_gate = nn.Sequential(
            nn.Linear(self.wdim, 1),
            nn.Sigmoid()
        )

        self.image_gate = nn.Sequential(
            nn.Linear(self.idim, 1),
            nn.Sigmoid()
        )

        self.word_encoder = nn.Sequential(
            nn.Linear(self.wdim, self.wdim1),
            nn.Tanh(),
            nn.Linear(self.wdim1, self.wdim2),
            nn.Tanh()
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(self.idim, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim2),
            nn.Tanh()
        )

        self.multi_encoder = nn.Sequential(
            nn.Linear(self.wdim2 + self.idim2, self.zdim),
            nn.Tanh()
        )

        self.multi_decoder = nn.Sequential(
            nn.Linear(self.zdim, self.wdim2 + self.idim2),
            nn.Tanh()
        )

        self.word_decoder = nn.Sequential(
            nn.Linear(self.wdim2, self.wdim1),
            nn.Tanh(),
            nn.Linear(self.wdim1, self.wdim),
            nn.Tanh()
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(self.idim2, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim),
        )

        self.init_parameters()

    def init_parameters(self):
        for net in [self.word_gate, self.image_gate,
                    self.word_encoder, self.word_decoder,
                    self.image_encoder, self.image_decoder,
                    self.multi_encoder, self.multi_decoder]:
            net.apply(init_weights)

    def forward(self, x_w, x_i):
        word_gate = self.word_gate(x_w).expand_as(x_w)
        image_gate = self.image_gate(x_i).expand_as(x_i)
        x_w = word_gate * x_w
        x_i = image_gate * x_i
        encoded_word = self.word_encoder(x_w)
        encoded_image = self.image_encoder(x_i)
        encoded = self.multi_encoder(torch.cat((encoded_word, encoded_image), dim=1))
        decoded = self.multi_decoder(encoded)
        decoded_word = self.word_decoder(decoded[:, :self.wdim2])
        decoded_image = self.image_decoder(decoded[:, -self.idim2:])
        return decoded_word, decoded_image, encoded
