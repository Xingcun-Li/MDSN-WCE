import torch
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

def get_weight(_train_dataset):
    dataloader = torch.utils.data.DataLoader(_train_dataset, batch_size=1)
    class_counts = [0 for i in range(len(_train_dataset.classes))]
    for _, label in dataloader:
        class_counts[label.item()] += 1
    # 计算每个类别的样本权重
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    
    return torch.FloatTensor(class_weights)


if __name__ == '__main__':
    train_dataset = ImageFolder(root='../data/mixedKID82/train', transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]))
    print(f"mean&std:{getStat(train_dataset)}\n\n")
    print("start")
    print(f"weight={get_weight(train_dataset)}\n")