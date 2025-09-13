python inference.py \
  --model_path "OmniGen2/OmniGen2" \
  --num_inference_step 50 \
  --image_guidance_scale 2.0 \
  --text_guidance_scale 4.0 \
  --instruction "inpaint the black part , with appropriate image" \
  --input_image_path example_images/inpaint4.png \
  --output_image_path outputs/raw_output.png \
  --num_images_per_prompt 1 \
  --transformer_lora_path experiments/ft_lora1752988566.034894/checkpoint-20000/transformer_lora