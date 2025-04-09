import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import wget
import os

model_url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.downsample(downsample)
        )
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.left(x)
        out = out + identity  
        out = F.relu(out)
        return out
    
class Bottleneck(nn.Module):
 
    expansion = 4
 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # -----------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        out = out + identity
        out = self.relu(out)
 
        return out

class ResNeXt(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 use_embedding=False
                ):
        super(ResNeXt, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.use_embedding=use_embedding
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
      
 
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
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
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            embedding = x
            x = self.fc(x)
        
        if self.use_embedding:   
            return x,embedding
        else:
            return x

class ResNeXt_mpa(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 embedding_size=2048,
                 beta=0.9,
                 temperature=0.2
                ):
        super(ResNeXt_mpa, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)    
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.beta = beta
        self.prototype = torch.zeros([self.num_classes,self.embedding_size],dtype=torch.float).cuda()
        self.temperature = temperature
      
 
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
 
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
        return nn.Sequential(*layers)
    
    @torch.no_grad()
    def update_memory(self, batch_embedding, category):
        batch_prototype = torch.zeros([self.num_classes,self.embedding_size]).cuda()
        count = torch.zeros([self.num_classes]).cuda()
        for i in range(category.size()[0]):
            batch_prototype[category[i]] += batch_embedding[i]
            count[category[i]]=count[category[i]]+1
        for i in range(self.num_classes):
            if count[i]!=0:
                batch_prototype[i] = batch_prototype[i]/count[i]
                self.prototype[i] = self.prototype[i] + self.beta * (batch_prototype[i] - self.prototype[i].detach())
    
    @torch.no_grad()
    def memory_prototype_attention(self, querys):
        # init output
        enhanced_querys = torch.zeros([querys.size()[0],self.embedding_size]).cuda()
        # per query for minibatch
        for i in range(querys.size()[0]):
            # weight
            query = querys[i].detach()
            attention_score = torch.zeros([self.prototype.size()[0]]).cuda()
            # compute cosine similarity between query and multiple prototype
            for j in range(self.prototype.size()[0]):
                attention_score[j] = torch.cosine_similarity(query.view(1, -1), self.prototype[j].view(1, -1).cuda())
                # attention_score[j] = torch.cosine_similarity(F.normalize(query.view(1, -1), p=2, dim=1), F.normalize(self.prototype[j].view(1, -1).cuda(),p=2,dim=1))
            # norm
            if torch.sum(attention_score)==0:
                continue

            attention_score = torch.exp(attention_score/self.temperature)/torch.sum(torch.exp(attention_score/self.temperature))
            attention_map = torch.zeros([self.embedding_size]).cuda()
            for k in range(self.prototype.size()[0]):
                attention_map = attention_map+attention_score[k].cuda()*self.prototype[k].cuda()
            # get enhanced query
            enhanced_querys[i] = attention_map
        return enhanced_querys
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            embedding = x
            x = x + self.memory_prototype_attention(querys=x)
            pae_embedding=x
            x = self.fc(x)
        return x,embedding #,pae_embedding


def ResNeXt50_32x4d(num_classes=1000, pretrained=True, include_top=True,use_mpa_embedding=False,use_embedding=False):
    
    groups = 32
    width_per_group = 4
    if use_mpa_embedding:
        model=ResNeXt_mpa(Bottleneck, [3, 4, 6, 3],
                          num_classes=num_classes,
                          include_top=include_top,
                          groups=groups,
                          width_per_group=width_per_group)
        
    else:
        model=ResNeXt(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group,
                      use_embedding=use_embedding)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, num_classes),
        nn.Softmax()
    )
    if pretrained:
        weight_dir="./model"
        weight_path = os.path.join(weight_dir, "resnext50_32x4d-1a0047aa.pth")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(weight_path):
            wget.download(model_url, weight_path)
        pretrain_weights = torch.load(weight_path,map_location=torch.device('cuda:0'))
        model_dict = model.state_dict()
        state_dict={k:v for k,v in pretrain_weights.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict,strict=False)
        
    return model


# no softmax, because need raw logit to perform logit adjustment
# def ResNeXt50_32x4d_logit(num_classes=1000, pretrained=True, include_top=True,use_mpa_embedding=False,use_embedding=False):
    
#     groups = 32
#     width_per_group = 4
#     if use_mpa_embedding:
#         model=ResNeXt_mpa(Bottleneck, [3, 4, 6, 3],
#                           num_classes=num_classes,
#                           include_top=include_top,
#                           groups=groups,
#                           width_per_group=width_per_group)
        
#     else:
#         model=ResNeXt(Bottleneck, [3, 4, 6, 3],
#                       num_classes=num_classes,
#                       include_top=include_top,
#                       groups=groups,
#                       width_per_group=width_per_group,
#                       use_embedding=use_embedding)
#     fc_inputs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(fc_inputs, 1024),
#         nn.ReLU(),
#         nn.Dropout(0.4),
#         nn.Linear(1024, num_classes)
#     )
#     if pretrained:
#         weight_dir="./model"
#         weight_path = os.path.join(weight_dir, "resnext50_32x4d-1a0047aa.pth")
#         if not os.path.exists(weight_dir):
#             os.makedirs(weight_dir)
#         if not os.path.exists(weight_path):
#             wget.download(model_url, weight_path)
#         pretrain_weights = torch.load(weight_path,map_location=torch.device('cuda:0'))
#         model_dict = model.state_dict()
#         state_dict={k:v for k,v in pretrain_weights.items() if k in model_dict.keys()}
#         model_dict.update(state_dict)
#         model.load_state_dict(model_dict,strict=False)
        
#     return model