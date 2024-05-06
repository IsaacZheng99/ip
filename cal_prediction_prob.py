import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from loader import RandomSubsetDataset


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="NN model")
    parser.add_argument("--size", type=eval, help="size of input image to the model")
    parser.add_argument("--mean", type=eval, help="mean for normalization")
    parser.add_argument("--std", type=eval, help="std for normalization")
    parser.add_argument("--batch_size", type=int, help="batch size for input images")
    parser.add_argument("--input_folder_path", type=str,
                        help="folder path of the input images")
    parser.add_argument("--input_data_type", type=str,
                        help="train, val, test")
    parser.add_argument("--ratio", type=float,
                        help="the ratio of images for generating HLTM, you can set 1 to use all the images")
    parser.add_argument("--sample_seed", type=int, help="seed for randomly selecting images")
    parser.add_argument("--output_path", type=str, help="path of file for recording the result")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set arguments
    args = set_arguments()

    # collect the input data
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize(args.mean, args.std)
    ])
    print(f"Loading images from {args.input_folder_path}.")
    # we use ImageNet dataset, so the directory structure is like "train/n01440764/*.jpeg"
    # please note the directory structure
    # you can change the way to reading your image files
    dataset = RandomSubsetDataset(
        args.input_folder_path,
        args.input_data_type,
        args.ratio,
        transform,
        args.sample_seed)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # calculate the prediction probabilities
    print(f"Calculate probabilities for {len(dataset.samples)} images.")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}.")
    model = models.resnet50(pretrained=True)
    model.to(device)
    model.eval()
    for batch_idx, (images, img_indices) in enumerate(data_loader):
        if batch_idx % 100 == 0:
            print(f"\tBatch: {batch_idx}")
        outputs = model(images.to(device))

        # record
        with open(args.output_path, "a") as file:
            for idx in range(len(img_indices)):
                probs = f"img{img_indices[idx].item()}"
                for output in outputs[idx]:
                    probs += f" {output.item()}"
                probs += "\n"
                file.write(probs)

    print("Finished.")
