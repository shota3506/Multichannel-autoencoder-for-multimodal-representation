import torch
import torch.nn as nn
from torch.nn import init


class AutoEncoder(nn.Module):
    def __init__(self, word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim):
        super(AutoEncoder, self).__init__()
        self.tdim = word_dim
        self.tdim1 = word_dim1
        self.tdim2 = word_dim2
        self.idim = image_dim
        self.idim1 = image_dim1
        self.idim2 = image_dim2
        self.zdim = multi_dim

        self.word_encoder = nn.Sequential(
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

        self.multi_encoder = nn.Sequential(
            nn.Linear(self.tdim2 + self.idim2, self.zdim),
            nn.Tanh()
        )

        self.multi_decoder = nn.Sequential(
            nn.Linear(self.zdim, self.tdim2 + self.idim2),
            nn.Tanh()
        )

        self.word_decoder = nn.Sequential(
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

        self.init_parameters()

    def init_parameters(self):
        init.kaiming_normal_(self.word_encoder[0].weight.data)
        init.kaiming_normal_(self.word_encoder[2].weight.data)
        init.constant_(self.word_encoder[0].bias.data, val=0)
        init.constant_(self.word_encoder[0].bias.data, val=0)

        init.kaiming_normal_(self.image_encoder[0].weight.data)
        init.kaiming_normal_(self.image_encoder[2].weight.data)
        init.constant_(self.image_encoder[0].bias.data, val=0)
        init.constant_(self.image_encoder[0].bias.data, val=0)

        init.kaiming_normal_(self.multi_encoder[0].weight.data)
        init.constant_(self.multi_encoder[0].bias.data, val=0)

        init.kaiming_normal_(self.multi_decoder[0].weight.data)
        init.constant_(self.multi_decoder[0].bias.data, val=0)

        init.kaiming_normal_(self.word_decoder[0].weight.data)
        init.kaiming_normal_(self.word_decoder[2].weight.data)
        init.constant_(self.word_decoder[0].bias.data, val=0)
        init.constant_(self.word_decoder[0].bias.data, val=0)

        init.kaiming_normal_(self.image_decoder[0].weight.data)
        init.kaiming_normal_(self.image_decoder[2].weight.data)
        init.constant_(self.image_decoder[0].bias.data, val=0)
        init.constant_(self.image_decoder[0].bias.data, val=0)

    def forward(self, x_t, x_i):
        encoded_word = self.word_encoder(x_t)
        encoded_image = self.image_encoder(x_i)
        encoded = self.multi_encoder(torch.cat((encoded_word, encoded_image), dim=1))
        decoded = self.multi_decoder(encoded)
        decoded_word = self.word_decoder(decoded[:, :self.tdim2])
        decoded_image = self.image_decoder(decoded[:, -self.idim2:])
        return decoded_word, decoded_image, encoded
