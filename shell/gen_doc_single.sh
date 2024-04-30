#CUDA_VISIBLE_DEVICES=0 \
python gen_doc_single.py \
       --model "resnet50(weights='ResNet50_Weights.DEFAULT')" \
       --size "(224, 224)" \
       --mean "(0.485, 0.456, 0.406)" \
       --std "(0.229, 0.224, 0.225)" \
       --batch_size 8 \
       --target_layer "layer4" \
       --presence_judge_way "mean" \
       --threshold 1.0 \
       --input_folder_path "./dataset/test" \
       --input_data_type "val" \
       --ratio 1.0 \
       --sample_seed 123 \
       --output_path "./result/document/test.txt" \
       --selected_image_indices_path "./result/document/test_selected.txt"
