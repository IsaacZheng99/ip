import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from loader import RandomSubsetDataset
from presence import PresenceJudge


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, help="number of GPUs")
    parser.add_argument("--model", type=str, help="NN model")
    parser.add_argument("--size", type=eval, help="size of input image to the model")
    parser.add_argument("--mean", type=eval, help="mean for normalization")
    parser.add_argument("--std", type=eval, help="std for normalization")
    parser.add_argument("--batch_size", type=int, help="batch size for input images")
    parser.add_argument("--target_layer", type=str, help="target layer for generating documents")
    parser.add_argument("--presence_judge_way", type=str,
                        help="way to judging the presence of neurons, so far optional: mean, max")
    parser.add_argument("--threshold", type=float, help="threshold for judging the presence of neurons")
    parser.add_argument("--input_folder_path", type=str,
                        help="folder path of the input images")
    parser.add_argument("--input_data_type", type=str,
                        help="train, val, test")
    parser.add_argument("--ratio", type=float,
                        help="the ratio of images for generating HLTM, you can set 1 to use all the images")
    parser.add_argument("--sample_seed", type=int, help="seed for randomly selecting images")
    parser.add_argument("--output_path", type=str, help="path of file for recording the result")
    parser.add_argument("--selected_image_indices_path", type=str, help="recorde selected image indices")
    args = parser.parse_args()
    return args


def set_distribution(rank, world_size):
    # set up for distributed computation
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distribute_dataloader(dataset, rank, world_size, batch_size, pin_memory=False, num_workers=0):
    # get distributed dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader


def cleanup():
    # clean up the process groups
    dist.destroy_process_group()


def main(rank, world_size, args):
    # set up for distributed computation
    set_distribution(rank, world_size)

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
    # get distributed dataloader
    data_loader = distribute_dataloader(dataset, rank, world_size, args.batch_size)

    # generate the documents
    print(f"Generating documents for {len(dataset.samples)} images.")
    print(f"Device: {rank}.")
    # model = eval("models." + args.model)
    model = models.resnet50(pretrained=True)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    presence_judge = PresenceJudge(
        model,
        args.target_layer,
        args.output_path,
        args.presence_judge_way,
        args.threshold
    )
    all_image_indices = []
    for batch_idx, (images, indices) in enumerate(data_loader):
        if batch_idx % 100 == 0:
            print(f"\tBatch: {batch_idx}")
        presence_judge.forward(images, indices)
        all_image_indices.extend(indices.tolist())

    # record the selected image indices
    print("Recording the selected image indices.")
    with open(args.selected_image_indices_path, "w") as file:
        for idx in all_image_indices:
            file.write(str(idx) + "\n")

    # clean up the process groups
    cleanup()

if __name__ == '__main__':
    # set arguments
    args = set_arguments()

    # distributed
    mp.spawn(
        main,
        args=(args.world_size, args),
        nprocs=args.world_size
    )

    print("Finished.")
