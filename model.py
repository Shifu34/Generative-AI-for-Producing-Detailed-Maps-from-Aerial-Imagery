from utils import *
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64):
        super(UNetGenerator, self).__init__()

        # Construct U-Net generator
        # Add the innermost layer
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)

        # Add intermediate layers with ngf * 8 filters
        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block)

        # Gradually reduce the number of filters
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)

        # Add the outermost layer
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

class UNetSkipConnectionBlock(nn.Module):
  #Defines a block of the U-Net model with skip connections.
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        #Sets input_nc to outer_nc if not explicitly provided.
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False) # A convolutional layer that halves the spatial resolution.
        downrelu = nn.LeakyReLU(0.2, True) #A LeakyReLU activation function with slope 0.2 for negative inputs.
        downnorm = nn.BatchNorm2d(inner_nc) # Batch normalization to stabilize training.
        uprelu = nn.ReLU(True) # A standard ReLU activation function for non-linearity.
        upnorm = nn.BatchNorm2d(outer_nc) #Batch normalization for the upsampled output.

        #andles the outermost layer of the U-Net
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: #Defines the innermost block of the U-Net.
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else: # Handles intermediate layers of the U-Net.
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up += [nn.Dropout(0.5)]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model) #Combines all the layers into a sequential module for efficient computation.

    def forward(self, x): #Defines how input data flows through the block
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
        


class NLayerDiscriminator(nn.Module):
  #Initializes the layers of the discriminator, defining a sequence of convolutional, batch normalization, and activation layers based on the number of downsampling layers (n_layers).
    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        """
        A PatchGAN discriminator
        Args:
            input_nc (int): Number of channels in input images. Since we concatenate input and target, input_nc= input_image + target_image channels
            ndf (int): Number of filters in the last conv layer
            n_layers (int): Number of downsampling layers
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        #Applies a convolution with stride 2 to downsample the input and extracts basic features.
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        #Adds a series of convolutional layers that progressively downsample the feature maps, increasing the number of filters to capture more complex spatial patterns.
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [ #Adds the last downsampling layer with a stride of 1 to retain spatial resolution while extracting deeper features.
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Final layer
        #Outputs a single-channel feature map where each element represents the likelihood of a patch being real or fake.        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

#Passes the input image through the sequential model (self.model), returning the discriminator's prediction for all patches in the input.
    def forward(self, x):
        return self.model(x)
