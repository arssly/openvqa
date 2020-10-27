import torch
from PIL import Image
from .resnet_model import model, preproc_transform


def preproc_to_feats(image):
    input_tensor = preproc_transform(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = model(input_batch).detach().squeeze().cpu()
        output = output.numpy().reshape((2048, -1)).transpose()
    return output


if __name__ == "__main__":
    output = preproc_to_feats(Image.open('/Users/macbook/Downloads/vqa/val2014/COCO_val2014_000000000042.jpg'))
    print ('output shape', output.shape)
