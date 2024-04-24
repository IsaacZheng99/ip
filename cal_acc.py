import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from hltm_analyze import HLTMNodesAnalyzer


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hltm_nodes_json_path", type=str, help="*.nodes.json file from HLTM")
    parser.add_argument("--class_index_path", type=str, help="class-index file of the dataset")
    parser.add_argument("--model", type=str, help="NN model")
    parser.add_argument("--size", type=eval, help="size of input image to the model")
    parser.add_argument("--mean", type=eval, help="mean for normalization")
    parser.add_argument("--std", type=eval, help="std for normalization")
    parser.add_argument("--batch_size", type=int, help="batch size for input images")
    parser.add_argument("--target_model_layer", type=str, help="add hook to the target layer of the model")
    parser.add_argument("--target_hltm_layer", type=int, help="calculate probability based on latent varibles of the target layer")
    parser.add_argument("--input_folder_path", type=str, help="folder path of the input images")
    parser.add_argument("--true_label", type=int, help="true label of the input images")
    parser.add_argument("--target_label", type=int, help="target label for calculating probability")

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
    image_count = len(all_images)

    # calculate accuracy for target class
    print("Calculating accuracy.")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}.")
    # model = eval("models." + args.model)
    model = models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    hook_handle = model._modules.get(args.target_model_layer).register_forward_hook(hook_feature)

    # calculate original accuracy
    with torch.no_grad():
        org_right = 0
        for images in data_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            org_right += (predicted == args.true_label).sum().item()
        print(f"\tOriginally, right: {org_right}, acc: {org_right / image_count:.5f}")

        print(f"\tTotal latent variables count: {len(hltm_nodes_analyzer[args.target_hltm_layer])}")
        for idx, latent_variable in enumerate(hltm_nodes_analyzer[args.target_hltm_layer], start=1):
            hook_handle.remove()
            dead_neurons = hltm_nodes_analyzer.var_neuron[latent_variable]
            model._modules.get(args.target_model_layer).register_forward_hook(hook_feature)
            right = 0
            for images in data_loader:
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                right += (predicted == args.true_label).sum().item()
            # here we simply consider latent variables that can change the accuracy
            # TODO maybe we can step forward to compare the change of top-k predicted classes
            if right != org_right:
                print(f"\t{idx}: latent variable: {latent_variable}, right: {right}, acc: {right / image_count:.5f}")

    print("Finished.")
