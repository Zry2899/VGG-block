import torch
import torch.nn as nn
import torch.nn.init as init


class VGG_BLOCK(nn.Module):
    """
    VGG_BLOCK:Conv2d -> BN -> ReLU
    支持自定义权重初始化(Xavier or Kaiming or default)
    """
    def __init__(self,in_channels,out_channels,num_convs=2,init_method='kaiming'):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            num_convs: 该block内卷积层数量
            init_method: 'kaiming', 'xavier', or 'default'
        """
        super().__init__()

        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        self.block = nn.Sequential(*layers)
        self.init_method = init_method
        self._initialize_weights()
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                if self.init_method=='kaiming':
                    init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                elif self.init_method=='xavier':
                    init.xavier_normal_(module.weight)
                else:
                    pass
            elif isinstance(module,nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    def forward(self,X):
        return self.block(X)
    
class VGG_NET(nn.Module):
    """
    Args:
        num_classes: the classes number
        init_method: the initial method of feature-layers(only involution layers)
    """
    def __init__(self,num_classes=10,init_method='kaiming'):
        super().__init__()
        in_channels = 3
        cfg = [64,128,256,512,512]
        num_convs_list = [2,2,3,3,3]
        layers = []
        for c,num_convs in zip(cfg,num_convs_list):
            layers.append(VGG_BLOCK(in_channels,c,num_convs,init_method))
            in_channels = c
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )
        self._init_classifier()
    def _init_classifier(self):
        for linear in self.classifier:
            if isinstance(linear,nn.Linear):
                init.kaiming_normal_(linear.weight)
                init.constant_(linear.bias,0)
    def forward(self,X):
        X = self.features(X)
        X = self.avgpool(X)
        X = torch.flatten(X,1)
        X = self.classifier(X)
        return X

if __name__=='__main__':
    from torchinfo import summary
    vgg_net = VGG_NET()
    summary(vgg_net,input_size=(1,3,224,224))





