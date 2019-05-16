import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
import argparse
from utils import *
from autoencoder import *


torch.manual_seed(1)


def evaluation(epoch, text, image, sound, autoencoder, vocab, args):
    testfile = ['men-3k.txt', 'simlex-999.txt', 'semsim.txt', 'vissim.txt', 'simverb-3500.txt',
                'wordsim353.txt', 'wordrel353.txt', 'association.dev.txt', 'association.dev.b.txt']

    _, _, _, multi_rep = autoencoder(text, image, sound)
    word_vecs = multi_rep.data.cpu().numpy()
    torch.save(autoencoder.state_dict(), open(args.outmodel + '.parameters-' + str(epoch), 'wb'))
    outfile = open(args.outmodel + '-' + str(epoch) + '.rep.txt', 'w')

    for i, w in enumerate(word_vecs):
        outfile.write(vocab[i] + ' ' + ' '.join([str(i) for i in w]) + '\n')

    for file in testfile:
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open('evaluation/' + file, 'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in vocab and word2 in vocab:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vecs[vocab.index(word1)],
                                                       word_vecs[vocab.index(word2)])
            else:
                not_found += 1
            total_size += 1
        sp = spearmans_rho(assingn_ranks(manual_dict), assingn_ranks(auto_dict))
        print(file)
        print("%15s" % str(total_size), "%15s" % str(not_found))
        print("%15.4f" % sp)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--text-dim', required=True, type=int)
    parser.add_argument('--image-dim', required=True, type=int)
    parser.add_argument('--sound-dim', required=True, type=int)
    parser.add_argument('--text-dim1', required=True, type=int)
    parser.add_argument('--text-dim2', required=True, type=int)
    parser.add_argument('--image-dim1', required=True, type=int)
    parser.add_argument('--image-dim2', required=True, type=int)
    parser.add_argument('--sound-dim1', required=True, type=int)
    parser.add_argument('--sound-dim2', required=True, type=int)
    parser.add_argument('--multi-dim', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--num-epoch', required=True, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--outmodel', required=True)
    parser.add_argument('--gpu', default=-1, type=int)
    args = parser.parse_args()

    indata = open(args.train_data)
    vocab = []
    text = []
    image = []
    sound = []
    for line in indata:
        line = line.strip().split()
        vocab.append(line[0])
        text.append(np.array([float(i) for i in line[1:args.text_dim+1]]))
        image.append(np.array([float(i) for i in line[args.text_dim+1:args.text_dim+args.image_dim+1]]))
        sound.append(np.array([float(i) for i in line[args.text_dim+args.image_dim+1:]]))
    text = torch.from_numpy(np.array(text)).type(torch.FloatTensor)
    image = torch.from_numpy(np.array(image)).type(torch.FloatTensor)
    sound = torch.from_numpy(np.array(sound)).type(torch.FloatTensor)
    train_indices = range(len(image))

    use_gpu = args.gpu > -1 and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    if use_gpu:
        train_loader = Data.DataLoader(dataset=train_indices, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    else:
        train_loader = Data.DataLoader(dataset=train_indices, batch_size=args.batch_size, shuffle=True)

    autoencoder = AutoEncoder(args).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    min_yloss = 99999

    total_text = Variable(text).to(device)
    total_image = Variable(image).to(device)
    total_sound = Variable(sound).to(device)

    for epoch in range(args.num_epoch):
        epoch += 1
        for step, indices in enumerate(train_loader):
            batch_text = Variable(text[indices].view(-1, args.text_dim)).to(device)
            batch_image = Variable(image[indices].view(-1, args.image_dim)).to(device)
            batch_sound = Variable(sound[indices].view(-1, args.sound_dim)).to(device)

            decoded_text, decoded_image, decoded_sound, _ = autoencoder(batch_text, batch_image, batch_sound)

            loss = loss_func(decoded_text, batch_text) \
                   + loss_func(decoded_image, batch_image) \
                   + loss_func(decoded_sound, batch_sound)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

        if epoch % 100 == 0:
            evaluation((epoch, total_text, total_image, total_sound, autoencoder, vocab, args))
