
python3 demo.py \
--input "jennifer_lawrence.jpg" \
--output "output" \
--detector-weights "pretrained/yolov8x_person_face.pt" \
--checkpoint "pretrained/checkpoint-377.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons
