from torchvision import transforms

# baseline
def get_baseline_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# adv augmentation
def get_advanced_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1)
        ),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1)
    ])

