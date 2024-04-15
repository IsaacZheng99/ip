#python gen_doc.py \
#       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
#       --size "(224, 224)" \
#       --mean "(0.485, 0.456, 0.406)" \
#       --std "(0.229, 0.224, 0.225)" \
#       --batch_size 64 \
#       --target_layer "layer4" \
#       --presence_judge_way "mean" \
#       --threshold 0.2 \
#       --input_folder_path "./dataset/imagenet/data/train" \
#       --ratio 0.5 \
#       --sample_seed 123 \
#       --output_path "./result/document/resnet50_layer4_train50_mean.txt" \
#       --selected_image_indices_path "./result/document/resnet50_layer4_val50_mean_selected.txt"

python gen_doc.py \
       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
       --size "(224, 224)" \
       --mean "(0.485, 0.456, 0.406)" \
       --std "(0.229, 0.224, 0.225)" \
       --batch_size 4 \
       --target_layer "layer4" \
       --presence_judge_way "max" \
       --threshold 5 \
       --input_folder_path "./dataset/val" \
       --ratio 1.0 \
       --sample_seed 123 \
       --output_path "./result/document/resnet50_layer4_val10000_max2.txt" \
       --selected_image_indices_path "./result/document/resnet50_layer4_val10000_max_selected2.txt"
