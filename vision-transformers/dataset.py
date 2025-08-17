import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset = datasets.Caltech101(root="./data", download=True, transform=transform)

TRAIN_SIZE = len(dataset) * 0.8
TEST_SIZE = len(dataset) - TRAIN_SIZE
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [TRAIN_SIZE, TEST_SIZE]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
