import os
from PIL import Image
import torch
from torch.nn import Module
from torchvision import transforms
from detectron2.checkpoint import DetectionCheckpointer  
from detectron2.config import get_cfg    
from detectron2.engine import default_setup  
from detectron2.modeling import build_model 
from .config import add_attribute_config

def setup():
	cfg = get_cfg()
	add_attribute_config(cfg)
	script_dir = os.path.dirname(__file__)
	cfg_path = os.path.join(script_dir, 'R-50-grid.yaml')
	cfg.merge_from_file(cfg_path)  
	# force the final residual block to have dilations 1   
	cfg.MODEL.RESNETS.RES5_DILATION = 1 
	cfg.freeze() 
	default_setup(cfg, [])
	return cfg


transf =  transforms.Compose([
    transforms.Resize((448,448)), 
    transforms.ToTensor(),
])

def preproc_transform(image): 
	image = image[:,:, [2,1,0]]
	image = transf(image)
	return {'width': 448, 'height': 448, 'image': torch.tensor(image)}


cfg = setup()
rcnn_model = build_model(cfg)
DetectionCheckpointer(rcnn_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

class GridFeat(Module):
	def __init__(self, rcnn): 
		super(GridFeat, self).__init__()
		self.rcnn = rcnn

	def forward(self, inputs):
		images = self.rcnn.preprocess_image(inputs) 
		features = self.rcnn.backbone(images.tensor) 
		outputs = self.rcnn.roi_heads.get_conv5_features(features) 
		outputs = torch.flatten(outputs, start_dim=-2)
		outputs = torch.transpose(outputs, 1, 2)
		return outputs

model = GridFeat(rcnn_model)



if __name__ == '__main__': 
	i = Image.open('/Users/macbook/Downloads/vqa/val2014/COCO_val2014_000000000042.jpg')
	i = preproc_transform(i)
	o = model(i)
	print('outputs shape is: ', o.shape)