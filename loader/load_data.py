import os
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class RandomSubsetDataset(Dataset):
    """
    Class RandomSubsetDataset serves as a dataset for randomly selecting input images.
    """
    def __init__(self,
                 input_folder_path: str,
                 ratio: float,
                 transform: transforms,
                 sample_seed: int):
        """
        :param input_folder_path: folder path of the input images
        :param ratio: the ratio of selected images
        :param transform: to transform images
        :param sample_seed: seed for randomly selecting images
        """
        self.input_folder_path = input_folder_path
        self.ratio = ratio
        self.transform = transform
        self.sample_seed = sample_seed
        self.samples = []  # todo [[img_path, img_idx]]
        self._select_samples()

    def _select_samples(self):
        random.seed(self.sample_seed)
        for sub_folder in os.listdir(self.input_folder_path):
            sub_folder_path = os.path.join(self.input_folder_path, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            image_paths = []
            image_indices = []
            for file_name in os.listdir(sub_folder_path):
                if not file_name.lower().endswith(".jpeg"):
                    continue
                image_path = os.path.join(sub_folder_path, file_name)
                image_paths.append(image_path)
                image_indices.append(int(file_name.split("_")[2].split(".")[0]))
            selected_count = min(int(len(image_paths) * self.ratio), len(image_paths))
            selected_lst_indices = random.sample(list(range(len(image_paths))), selected_count)
            selected_image_paths = [image_paths[idx] for idx in selected_lst_indices]
            selected_image_indices = [image_indices[idx] for idx in selected_lst_indices]
            for idx in range(len(selected_image_paths)):
                # self.samples.append((selected_image_paths[idx], sub_folder))
                self.samples.append((selected_image_paths[idx], selected_image_indices[idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        image = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
