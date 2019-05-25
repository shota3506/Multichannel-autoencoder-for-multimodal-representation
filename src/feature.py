import numpy as np
import torch
import os
from models.autoencoder import MultichannelAutoEncoder, GatedMultichannelAutoEncoder
from utils.utils import *


if __name__ == '__main__':
    word_file = os.path.join('..', 'data', 'glove.840B.300d', 'glove.840B.300d.txt')
    state_file = os.path.join('..', 'state', 'GatedMultichannelAutoEncoder.pth')
    result_file = os.path.join('..', 'result', 'GatedMultichannelAutoEncoder.txt')

    word_dim = 300
    word_dim1 = 200
    word_dim2 = 150
    image_dim = 512
    image_dim1 = 200
    image_dim2 = 150
    multi_dim = 300

    model = GatedMultichannelAutoEncoder(
        word_dim, word_dim1, word_dim2, image_dim, image_dim1, image_dim2, multi_dim
    )
    model.load_state_dict(torch.load(state_file))

    model.eval()
    with torch.no_grad():
        with open(word_file, 'r') as f:
            with open(result_file, 'w') as r:
                for i, l in enumerate(f):
                    word, word_vec = l.strip().split(' ', 1)
                    word_vec = np.array([word_vec.split()], dtype='float')
                    word_vec = torch.from_numpy(word_vec).type(torch.FloatTensor)
                    try:
                        _, multimodal_vec = model.feature(word_vec)
                        multimodal_vec = multimodal_vec.squeeze()
                        multimodal_vec = ' '.join(map(lambda x: str(round(x, 5)), multimodal_vec.tolist()))
                        r.write(word + ' ')
                        r.write(multimodal_vec + '\n')
                    except RuntimeError:
                        print("Error", word)
