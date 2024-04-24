import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from hltm_analyze import HLTMNodesAnalyzer


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hltm_nodes_json_path", type=str, help="*.nodes.json file from HLTM")
    parser.add_argument("--target_hltm_layer", type=int, help="analyze the target layer of HLTM")
    parser.add_argument("--class_index_path", type=str, help="class-index file of the dataset")
    parser.add_argument("--model", type=str, help="NN model")
    parser.add_argument("--size", type=eval, help="size of input image to the model")
    parser.add_argument("--mean", type=eval, help="mean for normalization")
    parser.add_argument("--std", type=eval, help="std for normalization")
    parser.add_argument("--batch_size", type=int, help="batch size for input images")
    parser.add_argument("--target_model_layer", type=str, help="add hook to the target layer of the model")
    parser.add_argument("--input_folder_path", type=str, help="folder path of the input images")
    parser.add_argument("--true_label", type=int, help="true label of the input images")
    parser.add_argument("--target_label", type=int, help="target label for calculating probability")
    parser.add_argument("--output_path", type=str, help="path of file for recording the result")

    args = parser.parse_args()
    return args


# TODO
dead_neurons = []
def hook_feature(module, input, output):
    for image_idx in range(output.shape[0]):
        for neuron_idx in range(output.shape[1]):
            # set dead neurons
            if neuron_idx in dead_neurons:
                output[image_idx][neuron_idx] = torch.zeros(output.shape[2:])


if __name__ == '__main__':
    # set arguments
    args = set_arguments()

    # create the HLTM nodes analyzer
    hltm_nodes_analyzer = HLTMNodesAnalyzer(args.hltm_nodes_json_path)
    hltm_nodes_analyzer.init()

    # collect the input data
    with open(args.class_index_path, 'r') as f:
        json_data = f.read()
        class_index = json.loads(json_data)
    image_folder = f"{args.input_folder_path}/{class_index[str(args.true_label)][0]}"
    print(f"Loading images from {image_folder}.")
    all_images = []
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])
    # we use ImageNet dataset, so the directory structure is like "train/n01440764/*.jpeg"
    # please note the directory structure
    # you can change the way to reading your image files
    for file_name in os.listdir(image_folder):
        if not file_name.lower().endswith(".jpeg"):
            continue
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        all_images.append(image)
    data_tensor = torch.stack(all_images, dim=0)
    data_loader = DataLoader(data_tensor, batch_size=args.batch_size, shuffle=False)

    # calculate probabilities for target class
    print("Calculating probabilities.")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}.")
    model = eval("models." + args.model)
    model.to(device)
    model.eval()
    hook_handle = model._modules.get(args.target_model_layer).register_forward_hook(hook_feature)

    # TODO
    latent_variables = ["org", "Z367", "Z2154"]
    latent_variables_count = len(latent_variables)
    probs = [[] for _ in range(latent_variables_count)]
    # for latent_variable in hltm_nodes_analyzer.layer_var[args.target_hltm_layer]:
    for idx in range(latent_variables_count):
        print(f"\tProcessing latent variable: {latent_variables[idx]}")
        hook_handle.remove()
        dead_neurons = hltm_nodes_analyzer.var_neuron.get(latent_variables[idx], [])
        model._modules.get(args.target_model_layer).register_forward_hook(hook_feature)
        # right = 0
        for images in data_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            # right += (predicted == args.true_label).sum().item()
            probs_outputs = torch.softmax(outputs.data, dim=1)
            top_values, top_indices = torch.topk(probs_outputs, k=3, dim=1)
            # print(top_values.tolist(), top_indices.tolist())
            probs[idx].extend(probs_outputs[:, args.target_label].tolist())

    # plot
    print("Plotting.")
    x = range(1, len(all_images) + 1)
    for idx in range(latent_variables_count):
        plt.plot(x, probs[idx], label=latent_variables[idx])
    plt.legend()
    plt.savefig(args.output_path)

    print("Finished.")
