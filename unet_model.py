import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # nn.Dropout(p=0.5),
        nn.BatchNorm2d(out_channels), 
        nn.ELU(alpha=1.0, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # nn.Dropout(p=0.5),
        nn.BatchNorm2d(out_channels), 
        nn.ELU(alpha=1.0, inplace=True)
    )   



class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(1, 64) 
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
               

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(p = 0.1)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(256 + 128, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.threshold = nn.Sigmoid()
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        out = self.threshold(out) 
        return out
def conv_factory(in_channels, out_channels):
  return nn.Sequential(
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, out_channels, 5, padding=4, dilation=2),
      nn.Dropout(p=0.5)
  )

def conv_act(in_channels, out_channels):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, padding=1),
      nn.ReLU(inplace=True)
  )

class DenseBlock(nn.Module):
  def __init__(self, in_channels, nlayer):
    super().__init__()
    self.conv_fac1 = conv_factory(in_channels, 16)
    self.conv_fac2 = conv_factory(in_channels+16, 16)
    self.conv_fac3 = conv_factory(in_channels+16*2, 16)
    self.conv_fac4 = conv_factory(in_channels+16*3, 16)
    self.nlayer = nlayer

  def forward(self, x):
    cv1 = self.conv_fac1(x)
    x = torch.cat([x, cv1], dim=1)

    cv2 = self.conv_fac2(x)
    x = torch.cat([x, cv2], dim=1)

    cv3 = self.conv_fac3(x)
    x = torch.cat([x, cv3], dim=1)

    if self.nlayer == 4:
      cv4 = self.conv_fac4(x)
      x = torch.cat([x, cv4], dim=1)
    
    return x

class DenseNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_down1 = conv_act(1, 64)
    self.dense1 = DenseBlock(64, 4)

    self.conv_down2 = conv_act(64+64, 128)
    self.dense2 = DenseBlock(128, 4)

    
    self.conv_down3 = conv_act(128+64, 256)
    self.dense3 = DenseBlock(256, 4)

    self.conv_down4 = conv_act(256+64, 512)
    self.dense4 = DenseBlock(512, 4)

    self.conv_down5 = conv_act(512+64, 1024)
    self.dense5 = DenseBlock(1024, 4)

    self.conv_up6 = conv_act(1024+64, 512)
    self.dense6 = DenseBlock(512, 3)
    self.conv_up62 = conv_act(512+48, 256)
    
    self.conv_up71 = conv_act(512+64, 256)
    self.dense7 = DenseBlock(256, 3)
    self.conv_up72 = conv_act(256+48, 128)

    self.conv_up81 = conv_act(256+64, 128)
    self.dense8 = DenseBlock(128, 3)
    self.conv_up82 = conv_act(128+48, 64)

    self.conv_up91 = conv_act(128+64, 64)
    self.dense9 = DenseBlock(64, 3)
    self.conv_up92 = conv_act(64+48, 32)
    
    self.conv_up10 = nn.Conv2d(32, 2, 3, padding=1)
    self.threshold = nn.Softmax(dim=1)

    self.maxpool = nn.MaxPool2d(2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
   

  def forward(self, x):
    conv1 = self.conv_down1(x) #64,256,256
    db1 = self.dense1(conv1) #128,256,256
    pool1 = self.maxpool(db1) #128,128,128

    conv2 = self.conv_down2(pool1) #128,128,128
    db2 = self.dense2(conv2) #192,128,128
    pool2 = self.maxpool(db2) #192,64,64

    conv3 = self.conv_down3(pool2) #256,64,64
    db3 = self.dense3(conv3) #320,64,64
    pool3 = self.maxpool(db3) #320,32,32

    conv4 = self.conv_down4(pool3) #512,32,32
    db4 = self.dense4(conv4) #576,32,32
    pool4 = self.maxpool(db4) #576,16,16

    conv5 = self.conv_down5(pool4) #1024,16,16
    db5 = self.dense5(conv5) #1088,16,16
    up5 = self.upsample(db5) #1088,32,32
    up5 = self.conv_up6(up5) #512,32,32
    merge5 = torch.cat([db4, up5], dim=1) #1088,32,32

    conv6 = self.conv_up6(merge5) #512,32,32
    db6 = self.dense6(conv6) #560,32,32
    up6 = self.upsample(db6) #560,64,64
    up6 = self.conv_up62(up6) #256,64,64
    merge6 = torch.cat([db3, up6], dim=1) #576,64,64

    conv7 = self.conv_up71(merge6) #256,64,64
    db7 = self.dense7(conv7) #304,64,64
    up7 = self.upsample(db7) #304,128,128
    up7 = self.conv_up72(up7) #128,128,128
    merge7 = torch.cat([db2, up7], dim=1) #320,128,128

    conv8 = self.conv_up81(merge7) #128,128,128
    db8 = self.dense8(conv8) #176,128,128
    up8 = self.upsample(db8) #176,256,256
    up8 = self.conv_up82(up8) #64,256,256
    merge8 = torch.cat([db1, up8], dim=1) #192,256,256

    conv9 = self.conv_up91(merge8) #64,256,256
    db9 = self.dense9(conv9) #112,256,256
    up9 = self.conv_up92(db9) #32,256,256
    
    conv10 = self.conv_up10(up9) #1,256,256
    output = self.threshold(conv10)
    
    return output
