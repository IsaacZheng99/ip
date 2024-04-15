#python gen_cam.py \
#       --hltm_nodes_json_path "./result/hltm/resnet50_layer4_train50_mean/resnet50_layer4_train50_mean.nodes.json" \
#       --class_index_path "./dataset/imagenet/label/imagenet_class_index.json" \
#       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
#       --size "(224, 224)" \
#       --mean "(0.485, 0.456, 0.406)" \
#       --std "(0.229, 0.224, 0.225)" \
#       --target_model_layer "layer4" \
#       --target_hltm_layer 5 \
#       --target_class_idx 330 \
#       --input_folder_path "./dataset/imagenet/data/val" \
#       --output_path "./result/cam/n02325366.jpeg"


python gen_cam.py \
       --hltm_nodes_json_path "./result/hltm/ResNet50-test10000/ResNet50-test10000.nodes.json" \
       --class_index_path "./dataset/imagenet/label/imagenet_class_index.json" \
       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
       --size "(224, 224)" \
       --mean "(0.485, 0.456, 0.406)" \
       --std "(0.229, 0.224, 0.225)" \
       --target_model_layer "layer4" \
       --target_hltm_layer 5 \
       --target_class_idx 331 \
       --true_label 330 \
       --input_folder_path "./dataset/imagenet/data/val_sep" \
       --output_path "./result/cam/val2.jpeg"
