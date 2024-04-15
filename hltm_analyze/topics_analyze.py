from typing import List, Dict
from collections import defaultdict, Counter
import json


class HLTMTopicsAnalyzer:
    """
    Class HLTMTopicsAnalyzer serves as an analyzer for the topics of HLTM.
    Specifically, these APIs correspond to the subroutine4 of HLTA.
    """
    def __init__(self,
                 label_path: str,
                 hltm_topics_json_path: str,
                 img_indices_path: str,
                 threshold: float):
        """
        :param label_path: true label for image dataset
        :param hltm_topics_json_path: json file path for "*.topics.json"
        :param img_indices_path: txt file path for input images
        :param threshold: posterior probability threshold
        """
        self.label_path = label_path
        self.topics_path: str = hltm_topics_json_path
        self.img_indices_path: str = img_indices_path
        self.threshold: float = threshold
        self.hltm_lst: List = []  # json file to list data
        self.labels: List = []  # true label for each image
        self.var_img: Dict = defaultdict(list)  # key: latent variable, value: [image index]
        self.var_class: Dict = defaultdict(list)  # key: latent variable, value: [class label]

    def init(self):
        """
        initialize HLTMTopicsAnalyzer
        """
        self.read_topic_file()  # read HLTM topic json file
        self.gen_var_img_dict()  # generate the dict with latent variable as key and image index as value
        self.rec_label_lst()  # record true label for each image
        self.gen_var_class_dict()  # generate the dict with latent variable as key and image label as value

    def read_topic_file(self):
        """
        read HLTM topic json file
        """
        with open(self.topics_path, 'r') as f:
            json_data = f.read()
        self.hltm_lst = json.loads(json_data)

    def rec_label_lst(self):
        """
        record true label for each image
        """
        # read input image indices file
        with open(self.img_indices_path, "r") as file:
            img_indices = file.readlines()
        img_indices = [int(idx.strip()) for idx in img_indices]
        # read true label file
        with open(self.label_path, "r") as file:
            labels = file.readlines()
        labels = [int(label.strip()) for label in labels]
        # record true label for each image
        for img_idx in img_indices:
            self.labels.append(labels[img_idx])

    def gen_var_img_dict(self):
        """
        generate the dict with latent variable as key and image index as value
        """
        for var_img_info in self.hltm_lst:
            for img_idx, post_prob in var_img_info["doc"]:
                if post_prob >= self.threshold:
                    self.var_img[var_img_info["topic"]].append(int(img_idx))

    def gen_var_class_dict(self):
        """
        generate the dict with latent variable as key and image label as value
        """
        for latent_variable, img_indices in self.var_img.items():
            img_labels = []
            for idx in img_indices:
                img_labels.append(self.labels[idx])
            label_counter = Counter(img_labels)
            self.var_class[latent_variable] = label_counter
