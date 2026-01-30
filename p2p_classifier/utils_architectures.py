import torch
import torch.nn as nn


##########################################################################################
##########################################################################################
##################################                      ##################################
##################################      GENERATOR       ##################################
##################################                      ##################################
##########################################################################################
##########################################################################################


# Define Basic Block for ResNet for generator with 1D convolutions
class BasicBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, odd_correction=False):
        super(BasicBlockGenerator, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2+stride, stride=stride, padding=1+odd_correction, output_padding=odd_correction*1)
        # self.conv1 = nn.Sequential(nn.Upsample(scale_factor=stride-odd_correction*1e-10, mode='linear'), nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0+odd_correction, bias=True, output_padding=odd_correction*1),
                # nn.Upsample(scale_factor=stride-odd_correction*1e-10, mode='linear'),
                # nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# Define ResNet Generator with convolutions
class ResNetGenerator(nn.Module):
    def __init__(self, block, layers, noise_dim=128, output_channels=6, autoencoder_mode=False):
        super(ResNetGenerator, self).__init__()
        self.autoencoder_mode = autoencoder_mode
        self.noise_dim = noise_dim
        self.in_channels = noise_dim
        self.output_channels = output_channels
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, odd_correction=True)
        self.layer3 = self._make_layer(block, 16, layers[2], stride=2, odd_correction=True)
        self.layer4 = self._make_layer(block, 16, layers[3], stride=2)
        self.conv1 = nn.ConvTranspose1d(16, output_channels, kernel_size=8, stride=2, padding=3, output_padding=0)
        # self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='linear'), nn.Conv1d(8, output_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'))
        self.encoder = ResNetDiscriminator(BasicBlockDiscriminator, [2, 2, 2, 2], encoder_mode=True)

    def _make_layer(self, block, out_channels, blocks, stride, odd_correction=False, extra_channels=0):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels+extra_channels, out_channels, stride, odd_correction=odd_correction))
            odd_correction = False
            extra_channels = 0
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, condition, label=None):
        x9, x8, x7, x6, x5, x4, x3, x2, x1 = self.encoder(condition)
        
        x8 = x8.view(-1, 2, self.noise_dim) # mu and logvar
        x = x8[:,0] + torch.exp(x8[:,1] / 2) * x # Reparameterization
        
        x = x.view(-1, self.noise_dim, 1) # (N, noise_dim, 1)
        x = x.expand(-1, -1, 6) # (N, noise_dim, 6)
        x = self.layer1(x) # (N, 64, 12)
        x = self.layer2(x) # (N, 32, 23)
        x = self.layer3(x) # (N, 16, 45)
        x = self.layer4(x) # (N, 16, 90)
        
        x = self.conv1(x) # (N, 6, 180)
        
        if self.autoencoder_mode: return x, x8
        return x


##########################################################################################
##########################################################################################
##################################                      ##################################
##################################    DISCRIMINATOR     ##################################
##################################                      ##################################
##########################################################################################
##########################################################################################


# Define Basic Block for ResNet for discriminator
class BasicBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.2):
        super(BasicBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True, padding_mode='reflect'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# Define ResNet for binary classification
class ResNetDiscriminator(nn.Module):
    def __init__(self, block, layers, dropout_prob=0.2, encoder_mode=False):
        super(ResNetDiscriminator, self).__init__()
        self.encoder_mode = encoder_mode
        self.in_channels = 16
        self.conv1 = nn.Conv1d(6, 16, kernel_size=7, stride=2, padding=3, padding_mode='reflect')
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 2*128, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2*128, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, out_channels, blocks, stride, dropout_prob=0.2):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x) # (N, 16, 90)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1) # (N, 16, 45)
        x3 = self.layer1(x2) # (N, 16, 45)
        x4 = self.layer2(x3) # (N, 32, 23)
        x5 = self.layer3(x4) # (N, 64, 12)
        x6 = self.layer4(x5) # (N, 2*128, 6)
        x7 = self.avgpool(x6) # (N, 2*128, 1)
        x8 = torch.flatten(x7, 1) # (N, 2*128)
        x8 = self.dropout(x8)
        x9 = self.fc(x8) # (N, 1)
        x9 = torch.sigmoid(x9)
        if self.encoder_mode: return x9, x8, x7, x6, x5, x4, x3, x2, x1
        return x9




##########################################################################################
##########################################################################################
##################################                      ##################################
##################################        RESNET        ##################################
##################################                      ##################################
##########################################################################################
##########################################################################################

# Define Basic Block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out
    
# Define Bottleneck Block for ResNet
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4, dropout_prob=0.5):
        super(Bottleneck, self).__init__()
        bottleneck_channels = out_channels // expansion
        
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# Define ResNet for binary classification or regression
class ResNet(nn.Module):
    def __init__(self, block, layers, dropout_prob=0.5, classifier=True, with_gyro=True, with_acc=True):
        super(ResNet, self).__init__()
        self.classifier = classifier
        self.in_channels = 64
        self.conv1 = nn.Conv1d(6, 64, kernel_size=7, stride=2, padding=3)
        if not with_gyro: self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3)
        if not with_acc: self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1) if classifier else nn.Linear(512, 2)
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, out_channels, blocks, stride, dropout_prob=0.5):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.classifier: x = torch.sigmoid(x)
        return x


##########################################################################################
##########################################################################################
##################################                      ##################################
##################################         CNN          ##################################
##################################                      ##################################
##########################################################################################
##########################################################################################

# Define the CNN architecture
class CNNClassifier(nn.Module):
    def __init__(self, compressed=0, dropout_prob=0.5, with_gyro=True, with_acc=True):
        super(CNNClassifier, self).__init__()
        self.compressed = compressed
        self.conv1 = nn.Conv1d(6, 32, kernel_size=3, stride=1, padding=1)
        if not with_gyro: self.conv1 = nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1)
        if not with_acc: self.conv1 = nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        if ~self.compressed: self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        if ~self.compressed: self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        if ~self.compressed: self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        if ~self.compressed: self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128*T, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = torch.relu(self.dropout(self.bn1(self.conv1(x))))
        if ~self.compressed: x = torch.relu(self.dropout(self.bn2(self.conv2(x))))
        x = torch.relu(self.dropout(self.bn3(self.conv3(x))))
        if ~self.compressed: x = torch.relu(self.dropout(self.bn4(self.conv4(x))))
        x = torch.relu(self.dropout(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc(x))
        return x

    
    
##########################################################################################
##########################################################################################
##################################                      ##################################
##################################       BioBank        ##################################
##################################                      ##################################
##########################################################################################
##########################################################################################
class BioBank(nn.Module):
    def __init__(self, model, outputs=1):
        super().__init__()
        device = next(model.parameters()).device
        self.feature_extractor = model.feature_extractor
        if outputs == 1:
            self.classifier = nn.Sequential(nn.Linear(512,32), nn.ReLU(),
                                            nn.Linear(32,outputs), nn.Sigmoid()).to(device)
        else:
            self.classifier = nn.Sequential(nn.Linear(512,32), nn.ReLU(),
                                            nn.Linear(32,outputs)).to(device)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x.flatten(1))
        return x