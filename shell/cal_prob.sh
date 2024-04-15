#python cal_prob.py \
#       --hltm_nodes_json_path "./result/hltm/resnet50_layer4_train50_mean/resnet50_layer4_train50_mean.nodes.json" \
#       --target_hltm_layer 5 \
#       --class_index_path "./dataset/imagenet/label/imagenet_class_index.json" \
#       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
#       --size "(224, 224)" \
#       --mean "(0.485, 0.456, 0.406)" \
#       --std "(0.229, 0.224, 0.225)" \
#       --batch_size 8 \
#       --target_model_layer "layer4" \
#       --input_folder_path "./dataset/imagenet/data/val" \
#       --true_label 331 \
#       --target_label 331 \
#       --output_path "./result/prob/hare.jpeg"

python cal_prob.py \
       --hltm_nodes_json_path "./result/hltm/ResNet50-test10000/ResNet50-test10000.nodes.json" \
       --target_hltm_layer 5 \
       --class_index_path "./dataset/imagenet/label/imagenet_class_index.json" \
       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
       --size "(224, 224)" \
       --mean "(0.485, 0.456, 0.406)" \
       --std "(0.229, 0.224, 0.225)" \
       --batch_size 8 \
       --target_model_layer "layer4" \
       --input_folder_path "./dataset/imagenet/data/val_sep" \
       --true_label 331 \
       --target_label 331 \
       --output_path "./result/prob/hare.jpeg"
