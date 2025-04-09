from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
import cv2
import numpy as np
import os
import torch
import cv2
import ttach as tta
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, 
            target_layer, reshape_transform)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):

        if self.cuda:
            input_tensor = input_tensor.cuda()
        
        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            output = output[0]
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category, 
            activations, grads, eigen_smooth)

        cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
        result = np.float32(result)
        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor,
            target_category, eigen_smooth)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, 
        reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))



        
# 1. load model
model = torch.load("./model/BEST_datasetCADCAP3classes811_testacc0.9836_f1score0.9837_mpaTriplet_modelResNext50_32x4d_bs64_seed42_epochs30_optimSGD_lr0.01_wd0.0.pt")
model.eval()
# print(model)

# 2.select which targetlayer for heatmap
target_layer = [model.layer4[-1].relu]
# target_layer = [model.layer4]
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 3. load image
image_name="AGD532" # 525
image_path = "./heatmap/"+image_name+".jpg"
rgb_img = plt.imread(image_path)
plt.imshow(rgb_img)
plt.show()
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.3663, 0.2485, 0.1391),
                            std =(0.2628, 0.1856, 0.1255))
    ])

# preprocess_image作用：归一化图像，并转成tensor
input_tensor = transform(rgb_img)
input_tensor = torch.tensor(np.expand_dims(input_tensor, 0))
rgb_img = cv2.resize(rgb_img, (224,224))
rgb_img = np.float32(rgb_img) / 255.0
plt.imshow(rgb_img)
plt.show()
mask_path = "./heatmap/"+image_name+"_a.jpg"
if os.path.exists(mask_path):
    # show the mask
    rgb_mask = plt.imread(mask_path)
    plt.imshow(rgb_mask)
    plt.show()

# create cam for generate grad-cam
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

target_category = None


# Get your cam
grayscale_cam = cam(input_tensor=input_tensor)  # [batch, 224,224]

input_tensor = input_tensor.to("cuda:0")
outputs,_,_ = model(input_tensor)
#----------------------------------
# 7. show heatmap on the top of original image
grayscale_cam = grayscale_cam[0]
print(f"grayscale_cam{grayscale_cam.shape}")
visualization = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite(f'./heatmap/{image_name}_HEATMAP.jpg', visualization)
grad_img = plt.imread(f"./heatmap/{image_name}_HEATMAP.jpg")
plt.imshow(grad_img)
plt.show()


# use seaborn for designed heatmap
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 加载原图
image = rgb_img  # 替换为你的原图路径

# 创建一个Figure对象和一个Axes对象
fig, ax = plt.subplots(figsize=(5, 5))  # 调整大小以适应热图和标尺

# 绘制热图
heatmap = sns.heatmap(grayscale_cam, cmap='cividis', ax=ax, alpha=0.5, cbar=False)
heatmap.set_clip_on(True)  # 超出尺寸部分将被裁剪

# 绘制原图
ax.imshow(rgb_img, alpha=0.5)
ax.axis('off')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig(f"./heatmap/{image_name}_heatmapSNS.png", dpi=600)

# 显示图形
plt.show()