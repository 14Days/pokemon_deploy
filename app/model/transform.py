from torchvision import transforms

# 归一化参数
_normal_mean = [0.4948052, 0.48568845, 0.44682974]
_normal_std = [0.24580306, 0.24236229, 0.2603115]

_normal_transform = transforms.Normalize(_normal_mean, _normal_std)

# 定义不同的transform
transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    _normal_transform
])
