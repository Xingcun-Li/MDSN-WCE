import argparse
import os
import logging
import numpy as np
import torch.optim as optim
import random
from torch.utils.data import random_split,Dataset,DataLoader
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
from resnext50 import ResNeXt50_32x4d
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import SNRDistance

torch.set_printoptions(sci_mode=False) # close sci_model like 0.001 instead of 1e-3

torch.autograd.set_detect_anomaly(True)
# get all args for the learning process
def get_args():
    parser = argparse.ArgumentParser(description="Args for mixedKID")
    parser.add_argument("--dataset",
                        type=str,
                        default="mixedKID",
                        help="DATASETNAME'")
    parser.add_argument("--datadir",
                        type=str,
                        default="../data/mixedKID822",
                        help="Dir of your dataset, like '/data/DATASETNAME'")
    parser.add_argument("--num_workers",
                        type=int,
                        default=12,
                        help="Number of workers for loading the data")
    parser.add_argument("--model",
                        type=str,
                        default="ResNext50_32x4d",
                        help="Model name, default=ResNeXt50_32x4d")
    parser.add_argument("--num_classes",
                        type=int,
                        default=3,
                        help="Number of classes, default: 3")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size, default=64")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed, default=42")
    parser.add_argument("--epochs",
                        type=int,
                        default=40,
                        help="Max training epochs, default=40")
    parser.add_argument("--optimizer",
                        type=str,
                        default="SGD",
                        help="optimizer, default=SGD")
    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.0000,
                        help="weight-decay (default: 0.0000)")
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="input visible devices for training (default: cuda:0)")
    args = parser.parse_args(args=[])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# document the info about the learning process on file
def get_log(file_name):
    logger = logging.getLogger('Image classification for wireless capsule endoscopy disease')
    logger.setLevel(logging.INFO)  # setting of log lever
    sh = logging.StreamHandler()  # output in terminal
    fh = logging.FileHandler(file_name, mode='w')  # log file output setting
    sh.setLevel(logging.INFO)  # set the level of handler
    fh.setLevel(logging.INFO) # 
    formatter = logging.Formatter('[%(asctime)s]\t%(message)s',datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(formatter)  # set format for output
    fh.setFormatter(formatter)
    if not logger.handlers: # to avoid multiple output
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


class TrDataset(Dataset):
    def __init__(self, base_dataset, transforms): 
        super(TrDataset, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transforms(x), y


def get_weight(_train_dataset, _args):
    dataloader = DataLoader(_train_dataset, batch_size=1)
    class_counts = [0 for i in range(_args.num_classes)]
    for _, label in dataloader:
        class_counts[label.item()] += 1
    # 计算每个类别的样本权重
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    
    return torch.FloatTensor(class_weights)


def load_data(_args):
    dataset_dir = _args.datadir
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')
    logger.info(torch.cuda.is_available())
    
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.TrivialAugmentWide(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3929, 0.2581, 0.1614),
                                 std =(0.2889, 0.2045, 0.1396))
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3929, 0.2581, 0.1614),
                                 std =(0.2889, 0.2045, 0.1396))
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.3929, 0.2581, 0.1614),
                                 std =(0.2889, 0.2045, 0.1396))
        ])
    }
    
    
    train_dataset=datasets.ImageFolder(train_dir,image_transforms['train'])
    valid_dataset=datasets.ImageFolder(valid_dir,image_transforms['valid'])
    test_dataset=datasets.ImageFolder(test_dir,image_transforms['test'])
    
    
    train_loader = DataLoader(train_dataset, batch_size=_args.batch_size, num_workers=_args.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=_args.batch_size, num_workers=_args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=_args.batch_size, num_workers=_args.num_workers, shuffle=False)

    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)
    test_data_size = len(test_dataset)
    
    logger.info(train_dataset.class_to_idx)
    logger.info(f"train_data_size = {train_data_size}, valid_data_size = {valid_data_size}, test_data_size = {test_data_size}")
    
    # class_weight = get_weight(train_dataset, _args)
    
    return train_loader, valid_loader, test_loader, train_data_size, valid_data_size, test_data_size


# load model by the model name
def get_model(_args):
    if _args.model=="ResNext50_32x4d":
        model = ResNeXt50_32x4d(num_classes=_args.num_classes,use_mpa_embedding=True)
    elif _args.model=="GCN":
        model = models.resnext50_32x4d(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, _args.num_classes),
            nn.Softmax()
        )
    else:
        model = models.resnext50_32x4d(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, _args.num_classes),
            nn.Softmax()
        )
    
    return model


def get_criterion(_weight,_args):
    return nn.CrossEntropyLoss(weight=_weight)


def get_optimizer(_model,_args):
    if _args.optimizer=="SGD":
        _optimizer = optim.SGD(_model.parameters(), lr=_args.lr, momentum=0.9, weight_decay=_args.weight_decay)
    else:
        _optimizer = optim.Adam(_model.parameters(), lr=_args.lr/10)
    return _optimizer


def get_scheduler(_optimizer, _T_max):
    return torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=_T_max)


class ConfusionMatrix(object):
    def __init__(self, num_classes: int):
        self.matrix = np.zeros((num_classes, num_classes))#初始化混淆矩阵，元素都为0
        self.num_classes = num_classes#类别数量

    def update(self, preds, labels):
        for p, t in zip(preds, labels):#pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1#根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):#计算指标函数
        logger.info(self.matrix)
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n#总体准确率
        logger.info(f"the model accuracy is {acc}")

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        #print("the model kappa is ", kappa)
        
        # precision, recall, specificity
        table = PrettyTable()#创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):#精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.#每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row(['class-'+str(i), Precision, Recall, Specificity])
        logger.info(table)
        return str(acc)

    def plot(self):#绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc='+self.summary()+')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def train(_train_loader, _valid_loader, _test_loader, _model, _criterion, _optimizer, _scheduler, _args):
    history = []
    best_acc = 0.0
    best_epoch = 0
    count = 0
    best_f1_score=0.0
    best_acc_path=""
    best_f1_path=""
    miner = miners.TripletMarginMiner(margin=1.0, type_of_triplets="semihard", distance=SNRDistance())
    loss_func = losses.TripletMarginLoss(margin=3.0,triplets_per_anchor="semihard", distance=SNRDistance())
    for epoch in range(_args.epochs):
        epoch_start = time.time()
        logger.info("\n\n\n************************Epoch: {}/{}***********************".format(epoch + 1, _args.epochs))
        _model.train()
        
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs,targets) in enumerate(tqdm(_train_loader)):
            inputs = inputs.to(f"cuda:{_args.gpu}")
            targets = targets.to(f"cuda:{_args.gpu}")
            
            outputs,embeddings = _model(inputs)
            # print(f"outputs.size()={outputs.size()}, targets.size()={targets.size()}")
            semihard_pairs = miner(embeddings,targets)
            w1 = (_args.epochs-epoch)/_args.epochs
            w2 = 1-w1
            loss1 = loss_func(embeddings,targets,semihard_pairs)
            loss2 = _criterion(outputs,targets)
            loss = w1*loss1 + w2*loss2
            # logger.info(f"triplet learning loss1 = {loss1}, CE loss2 = {loss2}, weight of loss1 loss2={w1 , w2}")
            _model.update_memory(embeddings,targets)
            
            
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(targets.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        confusion = ConfusionMatrix(num_classes=_args.num_classes)
        
        with torch.no_grad():
            _model.eval()
            prob_all = []
            label_all = []
            prob_all2 = []
            label_all2 = []
            for j, (inputs, targets) in enumerate(tqdm(_valid_loader)):
                inputs = inputs.to(f"cuda:{_args.gpu}")
                targets = targets.to(f"cuda:{_args.gpu}")
                outputs,embeddings = _model(inputs)
                # print(f"embeddings.size={embeddings.size()} \n\n\n targets.size={targets.size()}\n\n _model.prototype={_model.prototype}")
                loss = _criterion(outputs,targets)
                
                valid_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(targets.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)
                confusion.update(predictions.cpu().numpy(), targets.cpu().numpy())
                prob_all.extend(outputs.data.cpu().numpy())
                prob_all2.extend(predictions.cpu().numpy())
                label_all.extend(targets.cpu().numpy())
                label_all2.extend(targets.cpu().numpy())
            confusion.summary()
            F1_score=f1_score(label_all2,prob_all2,average='macro')
            logger.info("valid-macro-F1-Score:{:.4f}".format(F1_score))
            
        _scheduler.step()
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            
        if best_f1_score<F1_score:
            best_f1_score=F1_score
            best_f1_epoch = epoch+1

        epoch_end = time.time()

        logger.info("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f} %, \n\t\tValidation: Loss: {:.4f}, Accuracy:{:.4f} %, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        logger.info("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        logger.info("Best F1-score for validation : {:.4f} at epoch {:03d}".format(best_f1_score, best_f1_epoch))
        logger.info(f"learning rate = {_optimizer.state_dict()['param_groups'][0]['lr']}")


        

        
        if best_epoch==epoch+1:
            if os.path.exists(best_acc_path):
                os.remove(best_acc_path)
            best_acc_path = "./model/dataset{}{}classes{}_valacc{:.4f}_F1score{:.4f}_mpaTriplet_model{}_bs{}_seed{}_epoch{}_epochs{}_optim{}_lr{}_wd{}_ACC.pt".format(_args.dataset,_args.num_classes,_args.datadir[-3:],avg_valid_acc,F1_score,_args.model,_args.batch_size,_args.seed,str(epoch+1),_args.epochs,_args.optimizer,_args.lr,_args.weight_decay)
            torch.save(_model, best_acc_path)
        if best_f1_epoch==epoch+1:
            if os.path.exists(best_f1_path):
                os.remove(best_f1_path)
            best_f1_path="./model/dataset{}{}classes{}_valacc{:.4f}_F1score{:.4f}_mpaTriplet_model{}_bs{}_seed{}_epoch{}_epochs{}_optim{}_lr{}_wd{}_F1_SCORE.pt".format(_args.dataset,_args.num_classes,_args.datadir[-3:],avg_valid_acc,F1_score,_args.model,
                                                                                                         _args.batch_size,_args.seed,str(epoch+1),_args.epochs,
                                                                                                         _args.optimizer,_args.lr,_args.weight_decay)
            torch.save(_model, best_f1_path)
    best_f1_model = torch.load(best_f1_path)
    best_acc_model = torch.load(best_acc_path)
    os.remove(best_acc_path)
    os.remove(best_f1_path)
    return history, best_f1_model, best_acc_model

def evaluate(_test_loader,_model,_criterion,_args):
    logger.info("\n\n\nStart test process")
    test_start = time.time()
    test_loss = 0.0
    test_acc = 0.0
    confusion = ConfusionMatrix(num_classes=_args.num_classes)
    with torch.no_grad():
        _model.eval()
        prob_all = []
        label_all = []
        prob_all2 = []
        label_all2 = []
        for i, (inputs, targets) in enumerate(tqdm(_test_loader)):
            inputs = inputs.to(f"cuda:{_args.gpu}")
            targets = targets.to(f"cuda:{_args.gpu}")
            outputs,embeddings = _model(inputs)
            loss = _criterion(outputs,targets)
            test_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(targets.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * inputs.size(0)
            confusion.update(predictions.cpu().numpy(), targets.cpu().numpy())
            prob_all.extend(outputs.data.cpu().numpy())
            prob_all2.extend(predictions.cpu().numpy())
            label_all.extend(targets.cpu().numpy())
            label_all2.extend(targets.cpu().numpy())
        F1_score=f1_score(label_all2,prob_all2,average='macro')
        logger.info("test-macro-F1-Score:{:.4f}".format(F1_score))
        confusion.summary()
    avg_test_loss = test_loss / test_data_size
    avg_test_acc = test_acc / test_data_size
    test_end = time.time()
        

    logger.info("Test: Loss: {:.4f}, Accuracy: {:.4f} %, Time: {:.4f}s".format(avg_test_loss,
                                                                               avg_test_acc * 100,
                                                                               test_end - test_start))
    torch.save(_model, "./model/dataset{}{}classes{}_testacc{:.4f}_f1score{:.4f}_mpaTriplet_model{}_bs{}_seed{}_epochs{}_optim{}_lr{}_wd{}.pt".format(_args.dataset,
                                                                                                                                           _args.num_classes,
                                                                                                                                           _args.datadir[-3:],
                                                                                                                                           avg_test_acc,
                                                                                                                                           F1_score,
                                                                                                                                           _args.model,
                                                                                                                                           _args.batch_size,
                                                                                                                                           _args.seed,
                                                                                                                                           _args.epochs,
                                                                                                                                           _args.optimizer,
                                                                                                                                           _args.lr,
                                                                                                                                           _args.weight_decay)
              )
    


def main():
    #-------Load hyperparameters-------#
    args = get_args()
    setup_seed(seed=args.seed)
    log_path = "./log/dataset{}{}classes_model{}_bs{}_seed{}_epochs{}_optim{}_lr{}_wd{}.log".format(args.dataset,
                                                                                                    args.num_classes,
                                                                                                    args.model,
                                                                                                    args.batch_size,
                                                                                                    args.seed,
                                                                                                    args.epochs,
                                                                                                    args.optimizer,
                                                                                                    args.lr,
                                                                                                    args.weight_decay)
    global logger
    logger = get_log(log_path) 
    logger.info("KID: train information on KID dataset\n")
    logger.info("The List of Hyperparameters:")
    for k,v in sorted(vars(args).items()):
        logger.info("--{} = {}".format(k,v))
    
    #-------Load data-------#
    global train_data_size, valid_data_size, test_data_size
    train_loader, valid_loader, test_loader, train_data_size, valid_data_size, test_data_size = load_data(args)
    class_weight=torch.FloatTensor([3.6796, 2.3123, 3.3810])
    logger.info(f"class_weight = {class_weight}, open the function of 'get_weight' in load_data to compute class_weight, please save class_weight value and remove the time-consuming 'get_weight' from load_data")
    
    #-------Create model-------#
    model = get_model(args)
    
    #-------Create criterion and optimizer&scheduler-------#
    criterion = get_criterion(class_weight,args)
    optimizer = get_optimizer(model,args)
    scheduler = get_scheduler(optimizer,_T_max=5)
    
    #--------Use cuda------#
    model.to("cuda:{}".format(args.gpu))
    criterion.to("cuda:{}".format(args.gpu))
    
    #-------Training&validation-------#
    history,best_f1model,best_accmodel = train(train_loader, valid_loader, test_loader, model, criterion, optimizer, scheduler, args)
    logger.info("\n\n********************************The performance of best_f1model:*******************************")
    
    #-------Evaluation-------#
    evaluate(test_loader, best_f1model, criterion,args)
    logger.info("\n\n********************************The performance of best_accmodel:*******************************")
    evaluate(test_loader, best_accmodel, criterion,args)
    
if __name__ == "__main__":
    main()
