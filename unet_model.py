import torch
import torch.nn as nn

class Dense_Block(nn.Module):
  def __init__(self, in_channels,nb_layers,growth_rate):
    super(Dense_Block, self).__init__()
    # Batch Normalization
    self.bn = nn.BatchNorm2d(num_features = in_channels)
    # RELU
    self.relu = nn.ReLU(inplace = True)
    # Convolution
    self.conv1 = nn.Conv2d(in_channels, out_channels = growth_rate, kernel_size = (5,5), stride = 1, padding = (4,4), dilation=(2,2))
    # Dropout 
    self.drop=nn.Dropout(p=0.5)
  def forward(self, x, drop_out, nb_layers):
    # forward pass
    lst=[x]
    # Note: The Li model starts with a non-empty list to maintain the correct dimensions 
    for i in range(nb_layers):
      x1 = self.bn(x) 
      x1 = self.relu(x1)
      x1 = self.conv1(x1)
      if drop_out==True:
        x1 = self.drop(x1)
      else:
        x1=x1
      x_out= x1
      lst.append(x_out)
      # concatenate along the channel dimension
    x_final=torch.cat(lst, dim=1)
    return x_final   


######################################################################################    
#####################################################################################
####### ******************CONSTRUCT UP- CONVOLUTION FUNCTION*****************#########
#####################################################################################

class Up_Sample(nn.Module):
  def __init__(self, in_layers, out_layers):
    super(Up_Sample, self).__init__()
    # first upsample
    self.upsamp=nn.Upsample(scale_factor=2, mode='nearest')
    # then, put through convolution where the input and output layer numbers will be specified
    # when the function is called
    # NOTE: IN ORDER TO MAKE THE DIMENSIONS, WORK A DILATION RATE=2 WAS USED, this is the only divergence from the Li model
    self.conv=nn.Conv2d(in_layers, out_layers, kernel_size=(2,2), stride=1, padding = (1,1), dilation=(2, 2))
    self.relu=nn.ReLU(inplace = True)
  def forward(self, x):
    x = self.upsamp(x)
    x = self.conv(x)
    x = self.relu(x)
    return x

##################################################################################
##################################################################################
#************************CONSTRUCT UNET******************************************#
##################################################################################
##################################################################################
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
# THE REST OF THIS CODE BLOCK DEFINES THE ACTUAL NETWORK STRUCTURE... can be modified as needed
        # Begin encoding path...
        # PLACE DENSE BLOCKS IN ENCODING MODE
        self.conv1=nn.Conv2d(in_channels=1, out_channels = 64, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db1=Dense_Block(in_channels=64,nb_layers=4,growth_rate=16)
        self.pool1=torch.nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)
        self.conv2=nn.Conv2d(in_channels=128, out_channels = 128, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.db2=Dense_Block(in_channels=128,nb_layers=4,growth_rate=16)
        self.pool2=torch.nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)
        self.conv3=nn.Conv2d(in_channels=192, out_channels = 256, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db3=Dense_Block(in_channels=256,nb_layers=4,growth_rate=16)
        self.pool3=torch.nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)
        self.conv4=nn.Conv2d(in_channels=320, out_channels = 512, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db4=Dense_Block(in_channels=512,nb_layers=4,growth_rate=16)
        self.pool4=torch.nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)
        self.conv5=nn.Conv2d(in_channels=576, out_channels = 1024, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db5=Dense_Block(in_channels=1024,nb_layers=4,growth_rate=16)
        # ... encoding path complete
        # Begin decoding path...
        # PLACE DENSE BLOCKS IN DECODING MODE
        self.up5=Up_Sample(in_layers=1088, out_layers=512)
        #******** MERGE DEFINED IN FORWARD PASS ONLY 
        self.conv6=nn.Conv2d(in_channels=1088, out_channels = 512, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db6=Dense_Block(in_channels=512,nb_layers=3,growth_rate=16)
        self.up6=Up_Sample(in_layers=560, out_layers=256)
        #********* MERGE IN FORWARD PASS ONLY
        self.conv7=nn.Conv2d(in_channels=576, out_channels = 256, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db7=Dense_Block(in_channels=256,nb_layers=3,growth_rate=16)
        self.up7=Up_Sample(in_layers=304, out_layers=128)
        #********** MERGE IN FORARD PASS ONLY
        self.conv8=nn.Conv2d(in_channels=320, out_channels = 128, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db8=Dense_Block(in_channels=128,nb_layers=3,growth_rate=16)
        self.up8=Up_Sample(in_layers=176, out_layers=64)
        #********** MERGE IN FORARD PASS ONLY
        self.conv9=nn.Conv2d(in_channels=192, out_channels = 64, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)
        self.db9=Dense_Block(in_channels=64,nb_layers=3,growth_rate=16)
        self.conv10=nn.Conv2d(in_channels=112, out_channels = 32, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.relu=nn.ReLU(inplace = True)   
        # previously, out_channels=1
        self.conv11=nn.Conv2d(in_channels=32, out_channels = 1, kernel_size = (1,1), stride = 1, padding = (0,0))
        
    def forward(self, x):
        # begin encoding...
        C1 = self.conv1(x)
        R1 = self.relu(C1)
        DB1 = self.db1(R1, drop_out=False, nb_layers=4)
        P1 = self.pool1(DB1)
        C2 = self.conv2(P1)
        R2 = self.relu(C2)
        DB2 = self.db2(R2, drop_out=False, nb_layers=4)
        P2 = self.pool2(DB2)
        C3 = self.conv3(P2)
        R3 = self.relu(C3)
        DB3 = self.db3(R3, drop_out=False, nb_layers=4)
        P3 = self.pool3(DB3)
        C4 = self.conv4(P3)
        R4 = self.relu(C4)
        DB4 = self.db4(R4, drop_out=True, nb_layers=4)
        P4 = self.pool4(DB4)
        C5 = self.conv5(P4)
        R5 = self.relu(C5)
        DB5 = self.db5(R5, drop_out=True, nb_layers=4)
        #... encoding complete
        # Begin decoding...
        U5 = self.up5(DB5)
        # Merge 
        M5 = torch.cat((DB4, U5), dim=1)
        C6 = self.conv6(M5)
        R6 = self.relu(C6)
        DB6 = self.db6(R6, drop_out=False, nb_layers=3)
        U6 = self.up6(DB6)
        # Merge
        M6 = torch.cat((DB3, U6), dim=1)
        C7 = self.conv7(M6)
        R7 = self.relu(C7)
        DB7 = self.db7(R7, drop_out=False, nb_layers=3)
        U7 = self.up7(DB7)
        # Merge
        M7 = torch.cat((DB2, U7), dim=1)
        C8 = self.conv8(M7)
        R8 = self.relu(C8)
        DB8 = self.db8(R8, drop_out=False, nb_layers=3)
        U8 = self.up8(DB8)
        # Merge
        M8 = torch.cat((DB1, U8), dim=1)
        C9 = self.conv9(M8)
        R9 = self.relu(C9)
        DB9 = self.db9(R9, drop_out=False, nb_layers=3)
        C10 = self.conv10(DB9)
        R10 = self.relu(C10)
        C11 = self.conv11(R10)
        # Sigmoided added as final activation layer for classification for use with BCE Loss (calculated element-wise within the 3D tensors)
        OUT = torch.sigmoid(C11)
        return OUT 