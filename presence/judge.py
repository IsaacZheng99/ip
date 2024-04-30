from typing import Dict
from collections import defaultdict
import torch
import torch.nn as nn


class PresenceJudge:
    """
    Class PresenceJudge supports API for judging the presence/absence of neurons
    and generating the documents consists of neurons
    """
    def __init__(self,
                 model: nn.Module,
                 target_layer: str,
                 output_path: str,
                 presence_judge_way: str,
                 threshold: float):
        """
        :param model: model to use
        :param target_layer: target layer to generate GradCAM for
        :param output_path: write the documents to the output path
        :param presence_judge_way: way to judging the presence of neurons
        :param threshold: threshold for judging the presence of neurons
        """
        self.model: nn.Module = model
        self.model.eval()

        self.target_layer = target_layer
        self.output_path = output_path
        self.presence_judge_way = presence_judge_way
        self.presence_functions: Dict = {
            "mean": self.__mean_hook,
            "max": self.__max_hook,
            "wx": self.__wx_hook,
        }
        # register hook according to the presence way
        hook_func = self.presence_functions.get(self.presence_judge_way)
        if hook_func:
            # if torch.cuda.device_count() > 1:
            #     getattr(self.model.module, self.target_layer).register_forward_hook(hook_func)
            # else:
            #     getattr(self.model, self.target_layer).register_forward_hook(hook_func)
            getattr(self.model, self.target_layer).register_forward_hook(hook_func)
        self.threshold = threshold
        self.image_indices = None  # selected images indices

    def forward(self, images: torch.Tensor, indices: torch.Tensor) -> None:
        """
        :param images: batched input images
        :param indices: selected images indices
        """
        self.image_indices = indices.tolist()
        with torch.no_grad():
            outputs = self.model(images)

    def __mean_hook(self, module, input, output):
        batch_size = output.shape[0]
        neuron_count = output.shape[1]
        feature_maps = output.view(batch_size, neuron_count, -1)
        # get the mean value
        mean_maps = torch.mean(feature_maps, dim=2)
        present_neurons = defaultdict(list)  # key: image idx, value: [neuron]
        for image_idx in range(mean_maps.shape[0]):
            present_neurons[image_idx] = []
            for neuron_idx in range(mean_maps.shape[1]):
                # if mean value > threshold, we assume the neuron is present
                if mean_maps[image_idx][neuron_idx] >= self.threshold:
                    present_neurons[image_idx].append(neuron_idx)
        # convert the present neurons to present words
        all_documents = ""
        for image_idx, neurons in present_neurons.items():
            all_documents += f"{self.image_indices[image_idx]} "
            for neuron in neurons:
                all_documents += f"neuron{neuron} "
            all_documents += '\n'  # one line corresponds to one document
        with open(self.output_path, "a") as file:
            file.write(all_documents)

    def __max_hook(self, module, input, output):
        batch_size = output.shape[0]
        neuron_count = output.shape[1]
        feature_maps = output.view(batch_size, neuron_count, -1)
        # get the max value
        max_maps, _ = torch.max(feature_maps, dim=2)
        present_neurons = defaultdict(list)  # key: image idx, value: [neuron]
        for image_idx in range(max_maps.shape[0]):
            present_neurons[image_idx] = []
            for neuron_idx in range(max_maps.shape[1]):
                # if max value > threshold, we assume the neuron is present
                if max_maps[image_idx][neuron_idx] >= self.threshold:
                    present_neurons[image_idx].append(neuron_idx)
        # convert the present neurons to present words
        all_documents = ""
        for image_idx, neurons in present_neurons.items():
            all_documents += f"{self.image_indices[image_idx]} "
            for neuron in neurons:
                all_documents += f'neuron{neuron} '
            all_documents += '\n'  # one line corresponds to one document
        with open(self.output_path, "a") as file:
            file.write(all_documents)

    def __wx_hook(self, module, input, output):
        batch_size = output.shape[0]
        neuron_count = output.shape[1]
        feature_maps = output.view(batch_size, neuron_count, -1)
        # get the mean value (for ResNet50, there is an avgpool-layer)
        mean_maps = torch.mean(feature_maps, dim=2)
        present_neurons = defaultdict(list)  # key: image idx, value: [neuron]
        # TODO
        if self.target_layer == "layer4":
            for image_idx in range(mean_maps.shape[0]):
                present_neurons[image_idx] = []
                for neuron_idx in range(mean_maps.shape[1]):
                    # for layer4 of ResNet50, if wx value > threshold, we assume the neuron is present
                    max_wx = torch.max(mean_maps[image_idx][neuron_idx] * self.model.state_dict()['fc.weight'].T[neuron_idx])
                    if max_wx >= self.threshold:
                        present_neurons[image_idx].append(neuron_idx)
            # convert the present neurons to present words
            all_documents = ""
            for image_idx, neurons in present_neurons.items():
                all_documents += f"{self.image_indices[image_idx]} "
                for neuron in neurons:
                    all_documents += f"neuron{neuron} "
                all_documents += '\n'  # one line corresponds to one document
            with open(self.output_path, "a") as file:
                file.write(all_documents)