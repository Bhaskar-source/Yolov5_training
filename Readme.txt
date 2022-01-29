1. Environment create


[STEP -1] conda create --name yolov5-gpu python=3.6

[STEP -2] conda activate yolov5-gpu

[STEP -3] conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

[STEP -4] pip install -r requirements.txt




2. Training 

[STEP -1] Uplaod all images datasets/training_data/images && all labels datasets/training_data/labels

[STEP -2] cd datasets
	  python train_val.py
	  cd ..  

[STEP -3] change nc: # of classes  && names: all classes in list  in datasets/obj.yaml

[STEP -4] python train.py --weight ./weights/yolov5s.pt --data ./datasets/obj.yaml --cfg ./models/yolov5s.yaml --img-size 416 --batch-size 10 --epochs 10

 

