from typing import List, Dict
from collections import defaultdict
import json
import re


class HLTMNodesAnalyzer:
    """
    Class HLTMNodesAnalyzer serves as an analyzer for the nodes of HLTM.
    So far, this class pre-processes the HLTM and primarily get two dicts:
    1) self.layer_var: store latent variables according to layer;
    2) self.var_neuron: store neurons for each latent variable.
    """
    def __init__(self, hltm_nodes_json_path: str):
        """
        :param hltm_nodes_json_path: json file path for "*.nodes.json"
        """
        self.input_path: str = hltm_nodes_json_path
        self.hltm_lst: List = []  # json file to list data
        self.layer: int = -1  # layer of the HLTM
        self.layer_var: Dict = defaultdict(list)  # key: layer, value: [latent variable]
        self.var_neuron: Dict = defaultdict(list)  # key: latent variable, value: [neuron index]
        self.var_top_neuron: Dict = defaultdict(list)  # key: latent variable, value: [top_neuron index]

    def init(self):
        """
        initialize HLTMNodesAnalyzer
        """
        self.read_node_file()  # read HLTM node json file
        self.cal_layer()  # calculate the layer
        self.gen_layer_var_dict()  # generate the dict with layer as key and latent variable as value
        self.gen_var_neuron_dict()  # generate the dict with latent variable as key and neuron index as value
        self.gen_var_top_neuron_dict()  # generate the dict with latent variable as key and top_neuron (i.e., neurons in the "text" part) index as value

    def read_node_file(self):
        """
        read HLTM node json file
        """
        with open(self.input_path, 'r') as f:
            json_data = f.read()
        self.hltm_lst = json.loads(json_data)

    def cal_layer(self):
        """
        calculate the layer
        """
        if self.hltm_lst:
            self.layer = self.hltm_lst[0]["data"]["level"]

    def gen_layer_var_dict(self):
        """
        generate the dict with layer as key and latent variable as value
        """
        def traverse(cur_layer, cur_vars):
            if not cur_layer:
                return
            for var_dict in cur_vars:
                self.layer_var[cur_layer].append(var_dict["id"])
                traverse(cur_layer - 1, var_dict["children"])
        traverse(self.layer, self.hltm_lst)

    def gen_var_neuron_dict(self):
        """
        generate the dict with latent variable as key and neuron index as value
        """
        def backtrace(cur_layer, cur_vars):
            if not cur_layer:
                return []
            neurons = []
            for var_dict in cur_vars:
                indices = backtrace(cur_layer - 1, var_dict["children"])
                if not indices:  # for latent variables in layer1, just record "text" field
                    indices = re.findall(r'\d+', var_dict["text"])
                for index in indices:
                    self.var_neuron[var_dict["id"]].append(int(index))
                neurons.extend(indices)
            return neurons
        backtrace(self.layer, self.hltm_lst)

    def gen_var_top_neuron_dict(self):
        """
        generate the dict with latent variable as key and top_neuron (i.e., neurons in the "text" part) index as value
        """

        def traverse(cur_layer, cur_vars):
            if not cur_layer:
                return
            for var_dict in cur_vars:
                indices = re.findall(r'\d+', var_dict["text"])
                for index in indices:
                    self.var_top_neuron[var_dict["id"]].append(int(index))
                traverse(cur_layer - 1, var_dict["children"])
        traverse(self.layer, self.hltm_lst)
