import argparse
import matplotlib.pyplot as plt
from hltm_analyze import HLTMNodesAnalyzer, HLTMTopicsAnalyzer


"""
In this part, you need to use the subroutine4 of HLTM to generate the data, which stores in a json file.
Reference doc: https://github.com/kmpoon/hlta/tree/e498b56128cf1b1f59aa022bba009b13230f15b7?tab=readme-ov-file#subroutine-4-doc2vec-assignment
Reference code: https://github.com/kmpoon/hlta/blob/e498b56128cf1b1f59aa022bba009b13230f15b7/src/main/scala/tm/hlta/AssignTopics.scala#L27
"""


def set_arguments():
    """
    set arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, help="label for image dataset")
    parser.add_argument("--hltm_nodes_json_path", type=str, help="*.nodes.json file from HLTM")
    parser.add_argument("--hltm_tpoics_json_path", type=str, help="*.topics.json file from HLTM")
    parser.add_argument("--img_indices_path", type=str, help="indices for input images")
    parser.add_argument("--threshold", type=float, help="posterior probability threshold")
    parser.add_argument("--target_hltm_layer", type=int, help="target layer of HLTM")
    parser.add_argument("--class_num", type=int, help="number of classes of image dataset")
    parser.add_argument("--output_path", type=str, help="path of file for recording the result")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # set arguments
    args = set_arguments()

    # create the HLTM nodes analyzer
    hltm_nodes_analyzer = HLTMNodesAnalyzer(args.hltm_nodes_json_path)
    hltm_nodes_analyzer.init()

    # create the HLTM topics analyzer
    hltm_topics_analyzer = HLTMTopicsAnalyzer(
        args.label_path,
        args.hltm_tpoics_json_path,
        args.img_indices_path,
        args.threshold)
    hltm_topics_analyzer.init()

    # plot the distribution of classes that make p(Z|x) >= threshold
    # where Z is a latent variable, and x is an image
    print("Plotting.")
    print(f"\tHLTM layer: {args.target_hltm_layer}, "
          f"latent variables count: {len(hltm_nodes_analyzer.layer_var[args.target_hltm_layer])}")
    labels = range(1, args.class_num + 1)
    for latent_variable in hltm_nodes_analyzer.layer_var[args.target_hltm_layer]:
        if latent_variable in hltm_topics_analyzer.var_class:
            classes = hltm_topics_analyzer.var_class[latent_variable]
            class_count = [classes.get(label, 0) for label in labels]
            print(f"\tlatent variable: {latent_variable}, "
                  f"max number: {max(classes.values())}, "
                  f"label: {max(classes, key=classes.get)}")
            plt.plot(labels, class_count, label=latent_variable)
    plt.legend()
    plt.savefig(args.output_path)

    print("Finished.")
