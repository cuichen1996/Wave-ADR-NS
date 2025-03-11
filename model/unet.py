import torch
import torch.nn as nn
import torch.nn.functional as F

def getActivationFunction(
    act_function_name: str, features=None, end=False
):
    if act_function_name.lower() == "relu":
        return nn.ReLU()
    elif act_function_name.lower() == "celu":
        return nn.CELU(inplace=True)
    elif act_function_name.lower() == "relu_batchnorm":
        if end:
            return nn.ReLU(inplace=True)
        else:
            return nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(features))
    elif act_function_name.lower() == "tanh":
        return nn.Tanh()
    elif act_function_name.lower() == "elu":
        return nn.ELU()
    elif act_function_name.lower() == "prelu":
        return nn.PReLU()
    elif act_function_name.lower() == "gelu":
        return nn.GELU()
    elif act_function_name.lower() == "tanhshrink":
        return nn.Tanhshrink()
    elif act_function_name.lower() == "softplus":
        return nn.Softplus()
    elif act_function_name.lower() == "mish":
        return nn.Mish() 
    elif act_function_name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        err = "Unknown activation function {}".format(act_function_name)
        raise NotImplementedError(err)
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# %%
class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=5, padding=1),
            nn.BatchNorm2d(out_channels),
            getActivationFunction(act)
        )
        
    def forward(self, x):
        return self.Down(x)

# %%
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()

        self.up = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose2d(in_channels, out_channels, 5, 2, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

# %%
class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, act):
        super().__init__()

        self.up = nn.Sequential(
            getActivationFunction(act),
            nn.ConvTranspose2d(in_channels, out_channels, 5, 2, 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = self.up(x)
        return x

# %%
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act):
        super(ResNetBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            getActivationFunction(act),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        )
    
        self.activation = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            getActivationFunction(act)
        )
     
    def forward(self, x):
        out = self.layers(x) + x
        out = self.activation(out)
        return out

# %%
class UNetnoUP(nn.Module):
    def __init__(self, input_channel, kernel_size, act, M):
        super(UNetnoUP, self).__init__()
        self.act = act
        self.kernel_size= kernel_size

        self.conv_down_blocks = nn.ModuleList()
        self.conv_down_blocks.append(ConvDown(input_channel, 16, self.act))
        self.conv_down_blocks.append(ConvDown(16, 32, self.act))
        self.conv_down_blocks.append(ConvDown(32, 64, self.act))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(nn.AdaptiveAvgPool2d(8))
        self.ffcn = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64*8*8, 128),
                    getActivationFunction(self.act),
                    nn.Linear(128, M),
                    )

    def forward(self, x):
        x = self.conv_blocks[1](self.conv_blocks[0](self.conv_down_blocks[0](x)))
        x = self.conv_blocks[3](self.conv_blocks[2](self.conv_down_blocks[1](x)))
        x = self.conv_blocks[5](self.conv_blocks[4](self.conv_down_blocks[2](x)))
        x = self.conv_blocks[6](x)
        x = self.ffcn(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, act):
        super(UNet, self).__init__()
        self.act = act
        self.kernel_size= kernel_size
        self.mid_channel = 16+input_channel
        self.conv_down_blocks = nn.ModuleList()
        self.conv_down_blocks.append(ConvDown(input_channel, 16, self.act))
        self.conv_down_blocks.append(ConvDown(16, 32, self.act))
        self.conv_down_blocks.append(ConvDown(32, 64, self.act))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(16,16, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(64,64, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(32,32, self.kernel_size, self.act))
        self.conv_blocks.append(ResNetBlock(self.mid_channel,self.mid_channel, self.kernel_size, self.act))
        
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UNetUpBlock(64, 32, self.act))
        self.up_blocks.append(UNetUpBlock(64, 16, self.act))
        self.up_blocks.append(UNetUpBlock(32, 16, self.act))
        self.up_blocks.append(nn.Sequential(
            getActivationFunction(self.act),
            nn.Conv2d(self.mid_channel, output_channel, 3, 1, 1)
        ))

    def forward(self, x):
        # x = torch.cat([x, gridx, gridy], dim=1)
        op = self.conv_blocks[1](self.conv_blocks[0](self.conv_down_blocks[0](x)))
        x1 = self.conv_blocks[3](self.conv_blocks[2](self.conv_down_blocks[1](op)))
        x2 = self.conv_blocks[5](self.conv_blocks[4](self.conv_down_blocks[2](x1)))

        up_x3 = self.conv_blocks[6](x2)
        up_x3 = self.conv_blocks[7](up_x3)

        up_x1 = self.conv_blocks[8](self.up_blocks[0](up_x3, x1))
        up_x2 = self.conv_blocks[9](self.up_blocks[1](up_x1, op))
        up_x4 = self.conv_blocks[10](self.up_blocks[2](up_x2, x))
        y = self.up_blocks[3](up_x4)
        return y

class CNN(nn.Module):
    def __init__(self, n, cnn_layer_num, act):
        super(CNN, self).__init__()
        self.n = n
        self.activation = getActivationFunction(act)
      
        cnn_layers = []
        for i in range(cnn_layer_num):
            cnn_layers.append(nn.Conv2d(1 if i==0 else 32, 32, 3))
        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.cnn_layers.append(nn.AdaptiveMaxPool2d((n,n)))
        
        self.fc1 = nn.Linear(32*n*n+2, 128)
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, x,omega, N):
        for layer in self.cnn_layers:
            x = self.activation(layer(x)) 
        x = self.cnn_layers[-1](x)

        append_1 = torch.tensor([omega], dtype=torch.float, device=x.device) 
        append_2 = torch.tensor([N], dtype=torch.float, device=x.device)
        append_1 = append_1.repeat(x.shape[0], 1)
        append_2 = append_2.repeat(x.shape[0], 1)
        x = x.view(-1, 32 * self.n*self.n)  
        x = torch.cat([x, append_1, append_2], dim=1) 
        x = self.activation(self.fc1(x))  
        x = self.fc2(x)
        return x
    