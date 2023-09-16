import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path += ['/home/pratyush/Desktop/diffusion/models/autoencoders','..']
import h5py
from ae import autoencoder
import argparse
from dataset import Quark_Gluon_Dataset
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='autoencoder Testing')
parser.add_argument('--num_samples',type=int,default=10,help='training data size')
args = parser.parse_args()

path = '/home/pratyush/Desktop/diffusion'
data_path = os.path.join(path,'quark-gluon_data-set_n139306.hdf5')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = autoencoder().to(device)
model.load_state_dict(torch.load(os.path.join(path,'checkpoints/diffusion_ddpm.pt')))

# test_img = h5py.File(data_path,'r')['X_jets'][20000]
# test_img = (test_img - test_img.min())/(test_img.max()-test_img.min())

# test_img = torch.as_tensor(cv2.resize(test_img,(64,64))).unsqueeze(0)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                            torchvision.transforms.Lambda(lambda t: (t * 2) - 1)])
reverse_transforms = torchvision.transforms.Compose([
          torchvision.transforms.Lambda(lambda t: (t + 1) / 2),
           torchvision.transforms.Lambda(lambda t: t.permute(1,2,0)), # CHW to HWC
          torchvision.transforms.Lambda(lambda t: t * 255.),
          torchvision.transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
        #   torchvision.transforms.ToPILImage(),
      ])

data = Quark_Gluon_Dataset(data_path,num_samples=args.num_samples,transform=transform)
dataloader = torch.utils.data.DataLoader(data,batch_size=1,
                                         num_workers=2,shuffle=False)
jets,label = next(iter(dataloader))

reconstructed = model(jets.to(device))
reconstructed = reverse_transforms(reconstructed[0])

print(loss)
plt.subplot(2,2,1)
plt.imshow(reconstructed.permute(2,1,0).detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed_img')

plt.subplot(2,2,2)
plt.imshow(reconstructed[:,:,0].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed ECAL')

plt.subplot(2,2,3)
plt.imshow(reconstructed[:,:,1].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed HCAL')

plt.subplot(2,2,4)
plt.imshow(reconstructed[:,:,2].detach().cpu().numpy())
plt.axis('off')
plt.title('reconstructed Tracks')
plt.savefig(os.path.join(path,f'assets/ddpm_reconstructed_img_test.jpeg'))
plt.close()

plt.subplot(2,2,1)
plt.imshow(jets[0].permute(2,1,0).detach().cpu().numpy())
plt.axis('off')
plt.title('img')

plt.subplot(2,2,2)
plt.imshow(jets[:,:,0].detach().cpu().numpy())
plt.axis('off')
plt.title('ECAL')

plt.subplot(2,2,3)
plt.imshow(jets[:,:,1].detach().cpu().numpy())
plt.axis('off')
plt.title('HCAL')

plt.subplot(2,2,4)
plt.imshow(jets[:,:,2].detach().cpu().numpy())
plt.axis('off')
plt.title('Tracks')

plt.savefig(os.path.join(path,f'assets/ddpm_img_test.jpeg'))
plt.close()
