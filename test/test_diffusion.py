import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path += ['/home/pratyush/Desktop/diffusion/models/diffusion','..']
import h5py
from backward import UNet
from ddpm import DenoiseDiffusion_DDPM
import argparse
from dataset import Quark_Gluon_Dataset
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='Diffusion Testing')
parser.add_argument('--T',type=int,default=200,help='number of time steps T')
parser.add_argument('--num_samples',type=int,default=10,help='training data size')
args = parser.parse_args()

path = '/home/pratyush/Desktop/diffusion'
data_path = os.path.join(path,'quark-gluon_data-set_n139306.hdf5')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(dim=128,channels=3,dim_mults=(1,2,4,)).to(device)
model.load_state_dict(torch.load(os.path.join(path,'checkpoints/diffusion_ddpm.pt')))
diffusion = DenoiseDiffusion_DDPM(model,args.T,device)
# test_img = h5py.File(data_path,'r')['X_jets'][20000]
# test_img = (test_img - test_img.min())/(test_img.max()-test_img.min())

# test_img = torch.as_tensor(cv2.resize(test_img,(64,64))).unsqueeze(0)
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                            torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

data = Quark_Gluon_Dataset(data_path,num_samples=args.num_samples,transform=transform)
dataloader = torch.utils.data.DataLoader(data,batch_size=1,
                                         num_workers=2,shuffle=False)
jets,label = next(iter(dataloader))
print(jets.shape)
loss,reconstructed = diffusion.loss(jets.to(device))

print(loss)
plt.subplot(2,2,1)
plt.imshow(reconstructed[0].permute(2,1,0).detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed_img')

plt.subplot(2,2,2)
plt.imshow(reconstructed[0,0,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed ECAL')

plt.subplot(2,2,3)
plt.imshow(reconstructed[0,1,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed HCAL')

plt.subplot(2,2,4)
plt.imshow(reconstructed[0,2,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed Tracks')
plt.savefig(os.path.join(path,f'assets/ddpm_reconstructed_img_test.jpeg'))
plt.close()

plt.subplot(2,2,1)
plt.imshow(jets[0].permute(2,1,0).detach().cpu().numpy())
plt.axis('off')
plt.title('img')

plt.subplot(2,2,2)
plt.imshow(jets[0,0,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('ECAL')

plt.subplot(2,2,3)
plt.imshow(jets[0,1,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('HCAL')

plt.subplot(2,2,4)
plt.imshow(jets[0,2,:,:].detach().cpu().numpy())
plt.axis('off')
plt.title('Tracks')

plt.savefig(os.path.join(path,f'assets/ddpm_img_test.jpeg'))
plt.close()
