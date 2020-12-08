import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torchvision.transforms as transforms
import torch

from model import Net

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--modelname', type=str, default='tf_efficientnet_b4_ns', metavar='D',
                    help="modelname")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

state_dict = torch.load(args.model, map_location=torch.device('cpu'))
model = Net(args.modelname)
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

dict_modelname_size = {'tf_efficientnet_b4_ns' : 380, 'tf_efficientnet_b5_ns' : 456, 'swsl_resnext101_32x8d' : 224, 'ig_resnext101_32x32d' : 224, 'resnext101_32x8d' : 224}
size_ = dict_modelname_size[args.modelname]


data_transforms = transforms.Compose([
    transforms.Resize((size_, size_)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        

