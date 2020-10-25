import torch
from torchvision import transforms
from PIL import Image
from .resnest_model import resnest50

model = resnest50(pretrained=True)
model.eval()

preproc_transform = transforms.Compose([
    transforms.Resize((448,448)),  # TODO is this really 448 or 224 as said in docs?
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preproc_to_feats(image):
    input_tensor = preproc_transform(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    device_model = model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        device_model = model.to('cuda')
    with torch.no_grad():
        output = device_model(input_batch).detach().squeeze().cpu()
        output = output.numpy().reshape((2048, -1)).transpose()
    return output


if __name__ == "__main__":
    output = preproc_to_feats(Image.open('/Users/macbook/Downloads/vqa/val2014/COCO_val2014_000000000042.jpg'))
    print ('output shape', output.shape)

