
from pathlib import Path
import os
import numpy as np
from PIL import Image
from .resnet.preproc import preproc_to_feats


def preprocess_dataset(data_path = '/content/vqa',dest_path='/content/openvqa/data/vqa', preproc=preproc_to_feats):
	for p in Path(data_path).rglob('*.jpg'):
		image = Image.open(p)
		res = preproc(image)
		destination = os.path.join(dest_path, p[len(data_path), :]).replace('jpg', 'npz')
		np.savez(dest, res)
		os.unlink(p)