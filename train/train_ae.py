import torch
import torch.nn as nn
import torchvision
import sys
import os
import numpy as np
sys.path += ['/home/pratyush/Desktop/diffusion/models/autoencoders','..']
from dataset import Quark_Gluon_Dataset
from ae import autoencoder
import argparse
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='autoencoder Training')

parser.add_argument('--num_epochs',type=int,default=20,help = 'number of epochs')
parser.add_argument('--lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('--num_samples',type=int,default=10000,help='training data size')
parser.add_argument('--batch_size',type=int,default=32,help = 'batch size')

args = parser.parse_args()

#Dataset

path = '/home/pratyush/Desktop/diffusion'
os.makedirs(os.path.join(path,'testing'),exist_ok=True)
data_path = os.path.join(path,'quark-gluon_data-set_n139306.hdf5')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                            torchvision.transforms.Lambda(lambda t: (t * 2) - 1)])

reverse_transforms = torchvision.transforms.Compose([
          torchvision.transforms.Lambda(lambda t: (t + 1) / 2),
           torchvision.transforms.Lambda(lambda t: t.permute(1,2,0)), # CHW to HWC
          torchvision.transforms.Lambda(lambda t: t * 255.),
          torchvision.transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
      ]) 

data = Quark_Gluon_Dataset(data_path,num_samples=args.num_samples,transform=transform)
dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size,
                                         num_workers=2,shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = autoencoder().to(device)


optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
criterion = nn.MSELoss()
for epoch in range(args.num_epochs):
    total_loss = 0
    for step, (jets,labels) in enumerate(tqdm(dataloader,total = len(dataloader))):
        jets =jets.to(device)
        optimizer.zero_grad()
        output = model(jets)
        loss = criterion(output, jets)
        loss.backward()
        total_loss+=loss.item()
        optimizer.step()


        reconstructed = reverse_transforms(output[0])
        jets = reverse_transforms(jets[0])

        if (epoch+1) % 2 == 0 and step % 350 == 0:
            plt.subplot(2,2,1)
            plt.imshow(reconstructed)
            plt.axis('off')
            plt.title('reconstructed_img')

            plt.subplot(2,2,2)
            plt.imshow(reconstructed[:,:,0])
            plt.axis('off')
            plt.title('reconstructed ECAL')

            plt.subplot(2,2,3)
            plt.imshow(reconstructed[:,:,1])
            plt.axis('off')
            plt.title('reconstructed HCAL')

            plt.subplot(2,2,4)
            plt.imshow(reconstructed[:,:,2])
            plt.axis('off')
            plt.title('reconstructed Tracks')
            plt.savefig(os.path.join(path,f'testing/ddpm_reconstructed_img_epoch{epoch}_step{step}.jpeg'))
            plt.close()
            
            plt.subplot(2,2,1)
            plt.imshow(jets)
            plt.axis('off')
            plt.title('img')

            plt.subplot(2,2,2)
            plt.imshow(jets[:,:,0])
            plt.axis('off')
            plt.title('ECAL')

            plt.subplot(2,2,3)
            plt.imshow(jets[:,:,1])
            plt.axis('off')
            plt.title('HCAL')

            plt.subplot(2,2,4)
            plt.imshow(jets[:,:,2])
            plt.axis('off')
            plt.title('Tracks')

            plt.savefig(os.path.join(path,f'testing/ddpm_img_epoch{epoch}_step{step}.jpeg'))
            plt.close()
    torch.cuda.empty_cache()
    print(f'epoch: {epoch+1}, loss: {total_loss}')
torch.save(model.state_dict(),os.path.join(path,'checkpoints/diffusion_ddpm.pt'))