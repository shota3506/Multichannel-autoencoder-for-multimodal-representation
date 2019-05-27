import numpy as np
import torch
from torchtext import vocab
import os
from models.autoencoder import MultichannelAutoencoder, GatedMultichannelAutoencoder


if __name__ == '__main__':
    state_file = os.path.join('..', 'state', 'GatedMultichannelAutoencoder.pt')
    result_file = os.path.join('..', 'result', 'glove.300d.GatedMultichannelAutoencoder.pt')

    word_dim = 300
    word_dim1 = 200
    word_dim2 = 150
    image_dim = 512
    image_dim1 = 300
    image_dim2 = 150
    multi_dim = 300

    model = GatedMultichannelAutoencoder(
        word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim
    )
    model.load_state_dict(torch.load(state_file))

    glove = vocab.GloVe()

    vectors = {}

    model.eval()
    with torch.no_grad():
        for word in glove.stoi:
            _, multimodal_vec = model.feature(glove[word].unsqueeze(dim=0))
            multimodal_vec = multimodal_vec.squeeze()
            vectors[word] = multimodal_vec
            print(word)

    torch.save(vectors, result_file)
