
from pathlib import Path
import os
import numpy as np
from PIL import Image
from .resnet.preproc import preproc_to_feats
import time


def preprocess_dataset(data_path = '/content/vqa',dest_path='/content/openvqa/data/vqa', preproc=preproc_to_feats):
  start_time = time.time()
  c=0
  for p in Path(data_path).rglob('*.jpg'):
    c +=1
    if c % 100 == 0: 
      print('processed {} files'.format(c))
    image = Image.open(p).convert('RGB')
    res = preproc(image)
    destination = os.path.join(dest_path, p.relative_to(*Path(data_path).parts)).replace('jpg', 'npz')
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    np.savez(destination, res)
    os.unlink(p)
  print('time passed: ', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))


if __name__ == '__main__': 
  preprocess_dataset()