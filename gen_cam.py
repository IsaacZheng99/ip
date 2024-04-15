import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from cam import GradCAM
from hltm_analyze import HLTMNodesAnalyzer


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hltm_nodes_json_path", type=str,
                        help="*.nodes.json file from HLTM")
    parser.add_argument("--class_index_path", type=str, help="class-index file of the dataset")
    parser.add_argument("--model", type=str, help="NN model")
    parser.add_argument("--size", type=eval, help="size of input image to the model")
    parser.add_argument("--mean", type=eval, help="mean for normalization")
    parser.add_argument("--std", type=eval, help="std for normalization")
    parser.add_argument("--target_model_layer", type=str,
                        help="target layer of model for generating GradCAM plots")
    parser.add_argument("--target_hltm_layer", type=int,
                        help="target layer of HLTM for generating GradCAM plots")
    parser.add_argument("--target_class_idx", type=int,
                        help="target class index for generating GradCAM plots")
    parser.add_argument("--true_label", type=int, help="true label of the input images")
    parser.add_argument("--input_folder_path", type=str,
                        help="folder path of the input images")
    parser.add_argument("--output_path", type=str,
                        help="path of file for recording the result")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set arguments
    args = set_arguments()

    # create the HLTM nodes analyzer
    hltm_nodes_analyzer = HLTMNodesAnalyzer(args.hltm_nodes_json_path)
    hltm_nodes_analyzer.init()

    # collect images for generating GradCAM plots
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
    # we use ImageNet dataset, so the directory structure is like "val/n01440764/*.jpeg"
    # please note the directory structure
    # you can change the way to reading your image files
    for file_name in os.listdir(image_folder):
        if not file_name.lower().endswith(".jpeg"):
            continue
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path, 1)
        all_images.append(image)

    # generate the GradCAM plots
    print("Generating GradCAM plots.")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}.")
    model = eval("models." + args.model)
    model.to(device)
    model.eval()
    grad_cam = GradCAM(model, args.target_model_layer)
    all_cam_plots = []
    for image_idx in range(len(all_images)):
        image = all_images[image_idx]
        # Example: we use the top layer of the HLTM to generate GradCAM plots
        original_cam_plot = grad_cam.forward(image, transform, args.size, args.target_class_idx, (), device)
        all_cam_plots.append(original_cam_plot)
        # for latent_variable in ["Z367", "Z2154"]:
        for latent_variable in hltm_nodes_analyzer.layer_var[args.target_hltm_layer]:
            cam_plot = grad_cam.forward(image,
                                        transform,
                                        args.size,
                                        args.target_class_idx,
                                        hltm_nodes_analyzer.var_neuron[latent_variable], device)
            all_cam_plots.append(cam_plot)

    # plot
    print("Plotting.")
    fig, axs = plt.subplots(len(all_images), int(len(all_cam_plots) / len(all_images)))
    for idx, ax in enumerate(axs.flat):
        ax.imshow(all_cam_plots[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output_path)

    print("Finished.")
