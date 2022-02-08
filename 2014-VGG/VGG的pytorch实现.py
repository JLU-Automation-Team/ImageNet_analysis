import torch.nn as nn
import torch

# 定义VGG网络
# 损失函数CrossEntropyLoss中已经包含了Softmax函数，模型直接线性输出即可
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features #特征提取传入特征
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1) # 展平处理
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 模型参数的初始化
    def _initialize_weights(self):
        for m in self.modules():
            # 卷积层的参数初始化
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight) # 两种可选初始化方式
                # 网络截距项归零（每个像素卷完后再加bias）
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#初始化偏置为0
            # 全连接层的参数初始化
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01) # 两种可选初始化方式
                # 截距项归零
                nn.init.constant_(m.bias, 0)

# 特征提取层，即模型的卷积和池化层
def make_features(cfg: list):
    layers = [] #存放创建每一层结构
    in_channels = 3
    for v in cfg:
        # M代表下采样层
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 根据选择网络给定网络通道数
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

# 模型结构存放
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #数字为卷积层个数，M为池化层结构
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#实例化VGG网络
def vgg(model_name="vgg16", **kwargs): #**字典
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    finally:
        model = VGG(make_features(cfg), **kwargs)
    return model


