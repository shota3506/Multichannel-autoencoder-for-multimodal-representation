import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
import argparse
from utils import *

torch.manual_seed(1)


class AutoEncoder(nn.Module):
    def ___init__(self, args):
        super(AutoEncoder, self).__init__()
        self.tdim = args.text_dim
        self.tdim1 = args.text_dim1
        self.tdim2 = args.text_dim2
        self.idim = args.image_dim
        self.idim1 = args.image_dim1
        self.idim2 = args.image_dim2
        self.sdim = args.sound_dim
        self.sdim1 = args.sound_dim1
        self.sdim2 = args.sound_dim2
        self.zdim = args.multi_dim

        self.text_encoder = nn.Sequential(
            nn.Linear(self.tdim, self.tdim1),
            nn.Tanh(),
            nn.Linear(self.tdim1, self.tdim2),
            nn.Tanh()
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(self.idim, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim2),
            nn.Tanh()
        )

        self.sound_encoder = nn.Sequential(
            nn.Linear(self.sdim, self.sdim1),
            self.Tanh(),
            nn.Linear(self.sdim1, self.sdim2),
            nn.Tanh()
        )

        self.multi_encoder = nn.Sequential(
            nn.Linear(self.tdim2 + self.idim2 + self.sdim2, self.zdim),
            nn.Tanh()
        )

        self.multi_decoder = nn.Sequential(
            nn.Linear(self.zdim, self.tdim2 + self.idim2 + self.sdim2),
            nn.Tanh()
        )

        self.text_decoder = nn.Sequential(
            nn.Linear(self.tdim2, self.tdim1),
            nn.Tanh(),
            nn.Linear(self.tdim1, self.tdim),
            nn.Tanh()
        )

        self.image_decoder = nn.Sequential(
            nn.Linear(self.idim2, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim),
        )

        self.sound_decoder = nn.Sequential(
            nn.Linear(self.sdim2, self.sdim1),
            nn.Tanh(),
            nn.Linear(self.sdim1, self.sdim),
            nn.Tanh()
        )
        self.init_parameters()

    def init_parameters(self):
        init.kaiming_normal(self.text_encoder[0].weight.data)
        init.kaiming_normal(self.text_encoder[2].weight.data)
        init.constant(self.text_encoder[0].bias.data, val=0)
        init.constant(self.text_encoder[0].bias.data, val=0)

        init.kaiming_normal(self.image_encoder[0].weight.data)
        init.kaiming_normal(self.image_encoder[2].weight.data)
        init.constant(self.image_encoder[0].bias.data, val=0)
        init.constant(self.image_encoder[0].bias.data, val=0)

        init.kaiming_normal(self.sound_encoder[0].weight.data)
        init.kaiming_normal(self.sound_encoder[2].weight.data)
        init.constant(self.sound_encoder[0].bias.data, val=0)
        init.constant(self.sound_encoder[0].bias.data, val=0)

        init.kaiming_normal(self.multi_encoder[0].weight.data)
        init.constant(self.multi_encoder[0].bias.data, val=0)

        init.kaiming_normal(self.multi_decoder[0].weight.data)
        init.constant(self.multi_decoder[0].bias.data, val=0)

        init.kaiming_normal(self.text_decoder[0].weight.data)
        init.kaiming_normal(self.text_decoder[2].weight.data)
        init.constant(self.text_decoder[0].bias.data, val=0)
        init.constant(self.text_decoder[0].bias.data, val=0)

        init.kaiming_normal(self.image_decoder[0].weight.data)
        init.kaiming_normal(self.image_decoder[2].weight.data)
        init.constant(self.image_decoder[0].bias.data, val=0)
        init.constant(self.image_decoder[0].bias.data, val=0)

        init.kaiming_normal(self.sound_decoder[0].weight.data)
        init.kaiming_normal(self.sound_decoder[2].weight.data)
        init.constant(self.sound_decoder[0].bias.data, val=0)
        init.constant(self.sound_decoder[0].bias.data, val=0)

    def forward(self, x_t, x_i, x_s):
        encoded_text = self.text_encoder(x_t)
        encoded_image = self.image_encoder(x_i)
        encoded_sound = self.sound_encoder(x_s)
        encoded = self.multi_encoder(torch.cat((encoded_text, encoded_image, encoded_sound), dim=1))
        decoded = self.multi_decoder(encoded)
        decoded_text = self.text_decoder(decoded[:, 0:self.tdim2])
        decoded_image = self.image_decoder(decoded[:, self.tdim2:self.tdim2+self.idim2])
        decoded_sound = self.sound_decoder(decoded[:, self.tdim2+self.idim2:])
        return decoded_text, decoded_image, decoded_sound, encoded
