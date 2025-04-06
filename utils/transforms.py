from torchvision import transforms

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
