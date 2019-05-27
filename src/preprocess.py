import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchtext import vocab
from tqdm import tqdm
import os


if __name__ == '__main__':

    data_dir = '../data'

    image_data = os.path.join(data_dir, 'ILSVRC2012_img')
    image_classes = os.listdir(image_data)
    image_classes = [c for c in image_classes if os.path.isdir(os.path.join(image_data, c))]

    glove = vocab.GloVe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg = models.vgg19(pretrained=True).to(device)

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    vgg.eval()
    with torch.no_grad():
        with open(os.path.join(data_dir, 'glove.300d.vgg.512d.txt'), 'w') as f:
            pbar = tqdm(glove.stoi)
            for word in pbar:
                if word not in image_classes:
                    continue

                image_dir = os.path.join(image_data, word)
                image_datasets = datasets.ImageFolder(image_dir, image_transforms)
                image_dataloader = DataLoader(image_datasets, batch_size=len(image_datasets))

                for x, _ in image_dataloader:
                    x = x.to(device)
                    y = vgg.features(x)
                    y = y.mean(dim=(2, 3))

                    for image_vec in y:
                        word_vec = ' '.join(map(lambda x: str(round(x, 5)), glove[word].tolist()))
                        image_vec = ' '.join(map(lambda x: str(round(x, 5)), image_vec.tolist()))
                        f.write(word + ' ')
                        f.write(word_vec + ' ')
                        f.write(image_vec + '\n')
