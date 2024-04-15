#python analyze_posterior.py \
#       --label_path "./dataset/imagenet/label/ILSVRC2012_validation_ground_truth.txt" \
#       --hltm_nodes_json_path "./result/hltm/resnet50_layer4_train50_mean/resnet50_layer4_train50_mean.nodes.json" \
#       --hltm_tpoics_json_path "./result/hltm/resnet50_val_from_train50/resnet_val.topics.json" \
#       --img_indices_path "./result/document/resnet50_layer4_val50_mean_selected2.txt" \
#       --threshold 0.5 \
#       --target_hltm_layer 5 \
#       --class_num 1000 \
#       --output_path "./result/distribution/val250.jpeg"

CUDA_VISIBLE_DEVICES=0
python analyze_posterior.py \
       --label_path "./dataset/imagenet/label/ILSVRC2012_validation_ground_truth.txt" \
       --hltm_nodes_json_path "./result/hltm/ResNet50-test10000/ResNet50-test10000.nodes.json" \
       --hltm_tpoics_json_path "./result/hltm/ResNet50-val10000-from-test10000/ResNet50-val10000.topics.json" \
       --img_indices_path "./result/document/resnet50_layer4_val100_mean_selected.txt" \
       --threshold 0.5 \
       --target_hltm_layer 5 \
       --class_num 1000 \
       --output_path "./result/distribution/val1000.jpeg"
