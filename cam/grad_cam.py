from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from PIL import Image


class GradCAM:
    """
    Class GradCAM supports API of Gradient-based Class Activation Map
    Default values of attributes are for ImageNet dataset
    https://arxiv.org/abs/1610.02391
    """
    def __init__(self,
                 model: nn.Module,
                 target_layer: str,
                 num_cls: int = 1000):
        """
        :param model: model to use
        :param target_layer: target layer to generate GradCAM for
        :param num_cls: number of classes of the dataset (default 1000 for ImageNet dataset)
        """
        self.model: nn.Module = model
        self.model.eval()
        # register hook
        getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.model, target_layer).register_full_backward_hook(self.__backward_hook)
        self.num_cls: int = num_cls
        self.grads: List = []  # record gradients for generating GradCAM plots
        self.feature_maps: List = []  # record presence maps for generating GradCAM plots

    def forward(self,
                img: np.ndarray,
                transform: transforms.Compose,
                input_size: Tuple,
                class_idx: int,
                neuron_indices,
                device) -> np.ndarray:
        """
        backbone function for generating Grad-CAM
        :param img: input image
        :param transform: transform original image
        :param input_size: input size of the model
        :param class_idx: class to generate GradCAM for
        :param neuron_indices: indices of neurons to be mutated
        :param device: device to run
        :return: GradCAM plot
        """
        # pre-process image
        image_original_size = (img.shape[1], img.shape[0])
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        img_copy = Image.fromarray(np.uint8(img_copy))
        processed_img = transform(img_copy).unsqueeze(0)
        processed_img.to(device)

        # forward
        output = self.model(processed_img)
        idx = np.argmax(output.cpu().data.numpy())

        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, class_idx)
        loss.backward()

        # generate GradCAM
        grads = self.grads[0].cpu().data.numpy().squeeze()
        for neuron_idx in range(grads.shape[0]):
            # set dead neurons
            if neuron_indices and neuron_idx in neuron_indices:
            # if neuron_indices and neuron_idx not in neuron_indices:
                grads[neuron_idx] = torch.zeros(grads.shape[1:])
        fmap = self.feature_maps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads, input_size)

        # show cam
        cam_show = cv2.resize(cam, image_original_size)
        img_show = np.array(img).astype(np.float32) / 255
        cam_out = self.__show_cam_on_image(img_show, cam_show)
        self.feature_maps.clear()
        self.grads.clear()

        return cam_out

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.feature_maps.append(output)

    def __compute_loss(self, logit: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        compute loss for backpropagation
        :param logit: output of the model
        :param class_idx: class to generate GradCAM for
        :return: loss
        """
        if class_idx < 0 or class_idx >= 1000:
            # default for class with largest logit
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(class_idx)
        # note that the shape of logit is "torch.Size([1, 1000])"
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.num_cls).scatter_(1, index, 1)
        one_hot.requires_grad = True
        loss = torch.sum(one_hot * logit)
        return loss

    def __compute_cam(self, feature_map, grads, input_size) -> np.ndarray:
        """
        compute cam
        :param feature_map: presence map for a specific layer
        :param grads: gradient value for a specific layer
        :param input_size: input size of the model
        :return: cam
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        # global average pooling
        alpha = np.mean(grads, axis=(1, 2))
        # linear combination
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]
        # ReLU
        cam = np.maximum(cam, 0)
        # normalization
        cam = cv2.resize(cam, input_size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        return cam[:, :, ::-1]
