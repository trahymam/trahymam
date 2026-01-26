# %%
from PIL import Image
import os
import matplotlib.pyplot as plt
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import vgg19
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
def scale_down(img, scale):
    img = Image.open(img)
    d_img = img.resize((img.width//scale, img.height//scale), Image.BICUBIC)
    return d_img

# %%
def show_img(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title('low')
    ax[1].imshow(img2)
    ax[1].set_title('high')
    
    plt.show()

# %%
scale = 4
train_dir = 'train'
high_dir = os.path.join(train_dir, 'high')
images = os.listdir(high_dir)
l_images = os.listdir(os.path.join(train_dir, 'low'))
for i, image in enumerate(images):
    print(f'{i+1}')
    if image in l_images:
        continue
    u_img = os.path.join(high_dir, image)
    d_img = scale_down(u_img, scale)
    # show_img(d_img, Image.open(u_img))
    d_img.save(os.path.join(train_dir, 'low', image))
    
    torch.cuda.empty_cache()
    gc.collect()
    

# %%
# Adversarial Loss - Makes images look real
'''
Initially we give noise to generator, it tries to generate an image, it could be a random image. It goes to discriminator with real tag/classification(process says it's real not fake) along with real images. But when it compares generator fake image with real images, it finds out that it's fake. Mostly two type of problem we humans can see:
1. It doesn't look like real image
2. It doesn't look like target image
To make image more realistic we calculate adversarial loss. Adversarial(opposing) relationship between two networks - Generator and Discriminator. Generator tries to forge images to fool discriminator, discriminator tries to spot the forgery. Adversial loss is how often discriminator gets tricked. We tries to minimiae this loss. In this process generator learns to make images appearing more realistic. Now images looks more like real images. It's not guaranteed that it looks like the target image. Now the perceptual loss comes into play.
'''
# Perceptual Loss - Makes them look like the target image
'''
Now generator generates random realistic images which doesn't look like target image. Again discriminator can easily spot the forgery. Now we calcualate perceptual loss between generated image and target image. This mimic how humans perceive images, they doesn't match pixel by pixel but looks feature similarity. Think of an example "Human Seating", now your mind could easily draw an image. We just match that human is there and seating. We don't care how tall he/she is, where he/she is seating, what color cloth he/she is wearing. So perceptual loss captures high level features and tries to minimize the loss. It helps to generate images looking like target image.

Feature extraction training is costly so we use pre-trained models like VGG, ResNet etc.
'''


# %%
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_activation=True, use_batchnorm=True, **kwargs):
        super().__init__()
        self.use_activation = use_activation
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.ac = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        cnn = self.cnn(x)
        bn = self.bn(cnn)
        out = self.ac(bn) if self.use_activation else bn
        return out

# %%
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor**2, 2, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.ac = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x):
        return self.ac(self.ps(self.conv(x)))

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.b2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
        
    def forward(self, x):
        b1 = self.b1(x)
        return b1 + self.b2(b1)

# %%
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=7, stride=1, padding=4, use_batchnorm=False)
        self.res = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_activation=False)
        self.up = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=1)
    
    def forward(self, x):
        initial = self.initial(x)
        res = self.res(initial)
        conv = self.conv(res) + initial
        up = self.up(conv)
        out = self.final(up)
        return torch.sigmoid(out)

# %%
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for i, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, feature, kernel_size=3, stride = i % 2 + 1, padding=1, use_activation=True, use_batchnorm=i!=0))
            in_channels = feature
        
        self.blocks = nn.Sequential(*blocks)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(512*8*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        return self.mlp(self.blocks(x))

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4
epochs = 3
batch_size = 16
num_workers = 0
img_channels = 3

# %%
class vggL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights='DEFAULT').features[:25].eval().to(device)
        self.loss = nn.MSELoss()
    
    def forward(self, first, second):
        vgg_first = self.vgg(first)
        vgg_second = self.vgg(second)
        perceptual_loss = self.loss(vgg_first, vgg_second)
        return perceptual_loss

# %%
gen = Generator(in_channels=3).to(device)
disc = Discriminator(in_channels=3).to(device)
opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))
opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.999))
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = vggL()

# %%
transform_low = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

transform_high = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# %%
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        super(ImageDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        files_low = os.listdir(os.path.join(root_dir, 'low'))
        files_high = os.listdir(os.path.join(root_dir, 'high'))
        self.data = list(zip(files_low, files_high))
        print(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_low_file, img_high_file = self.data[index]
        low_res_path = os.path.join(self.root_dir, 'low', img_low_file)
        high_res_path = os.path.join(self.root_dir, 'high', img_high_file)

        low_res = np.array(Image.open(low_res_path))
        high_res = np.array(Image.open(high_res_path))
        
        if len(low_res.shape) != 3:
            low_res = np.stack([low_res] * 3, axis=-1)
        if len(high_res.shape) != 3:
            high_res = np.stack([high_res] * 3, axis=-1)
        low_res = low_res[:, :, :3]
        high_res = high_res[:, :, :3]
        
        low_res = transform_low(low_res)
        high_res = transform_high(high_res)
        
        return low_res, high_res

# %%
train_dataset = ImageDataset(root_dir = './train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

# %%
def train_fn(train_loader, gen, disc, opt_gen, opt_dis, bce, vggL):
    loop = tqdm(train_loader)
    print(f'Number of batches: {len(loop)}')
    disc_loss = 0
    gen_loss = 0
    
    for i, (low, high) in enumerate(loop):
        low = low.to(device)
        high = high.to(device)

        fake = gen(low)
        disc_real = disc(high)
        disc_fake = disc(fake.detach())
        
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        
        disc_loss = disc_loss_fake + disc_loss_real
        
        opt_dis.zero_grad()
        disc_loss.backward()
        opt_dis.step()
        
        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        vgg_loss = 0.006 * vggL(fake, high)
        gen_loss = vgg_loss + adversarial_loss
        
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        
    return gen_loss.detach().cpu(), disc_loss.detach().cpu()

# %%
d_losses = []
g_losses = []

for epoch in range(epochs):
    gc.collect()
    torch.cuda.empty_cache()
    print(f'Epoch:{epoch+1} / {epochs}')
    gen_loss, disc_loss = train_fn(train_loader, gen, disc, opt_gen, opt_disc, bce, vgg_loss)
    
    d_losses.append(disc_loss)
    g_losses.append(gen_loss)

# %%
torch.save(gen.state_dict(), 'checkpoint1_gen.pth')

# %%
# gen = Generator().to(device)
gen.load_state_dict(torch.load('checkpoint1_gen.pth', map_location=device))
gen.eval()

# %%
low_res_img = Image.open('test/low/img1.jpg')
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
low_res_tensor = preprocess(low_res_img).unsqueeze(0).to(device)

# %%


# %%
with torch.no_grad():
    high_res_tensor = gen(low_res_tensor)

high_res_image = transforms.ToPILImage()(high_res_tensor.squeeze(0).cpu())
high_res_image.save('test/high/img1.jpg')

# %%



