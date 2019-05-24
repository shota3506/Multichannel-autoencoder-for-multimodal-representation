import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from models.autoencoder import MultichannelAutoEncoder, GatedMultichannelAutoEncoder


class FeatureDataset(Dataset):
    def __init__(self, file, word_dim, image_dim, transform=None):
        self.transform = transform

        self.vocab = []
        self.word_vecs = []
        self.image_vecs = []
        self.size = 0

        with open(file, 'r') as f:
            for l in f:
                word, vecs = l.strip().split(' ', 1)
                vecs = vecs.split(' ')
                self.vocab.append(word)
                self.word_vecs.append(np.array([float(i) for i in vecs[:word_dim]]))
                self.image_vecs.append(np.array([float(i) for i in vecs[-image_dim:]]))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        word, word_vec, image_vec = self.vocab[idx], self.word_vecs[idx], self.image_vecs[idx]

        if self.transform:
            word_vec, image_vec = self.transform(word_vec), self.transform(image_vec)

        return word, word_vec, image_vec


if __name__ == '__main__':
    data_file = os.path.join('..', 'data', 'glove.300d.vgg.512d.txt')
    result_file = os.path.join('..', 'result', 'autoencoder.pth')
    batch_size = 64
    num_epochs = 300
    lr = 1e-3

    word_dim = 300
    word_dim1 = 200
    word_dim2 = 150
    image_dim = 512
    image_dim1 = 200
    image_dim2 = 150
    multi_dim = 300

    dataset = FeatureDataset(data_file, word_dim, image_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = GatedMultichannelAutoEncoder(word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):

        pbar = tqdm(dataloader)
        for i_batch, (word, batch_words, batch_images) in enumerate(pbar):
            pbar.set_description('epoch %3d / %d' % (epoch + 1, num_epochs))
            batch_words = batch_words.float().to(device)
            batch_images = batch_images.float().to(device)

            decoded_words, decoded_images, _ = autoencoder(batch_words, batch_images)

            loss = loss_func(decoded_words, batch_words) + loss_func(decoded_images, batch_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                pbar.set_postfix(loss=loss.item())

    torch.save(autoencoder.state_dict(), autoencoder.__class__.__name__ + '.pth')

