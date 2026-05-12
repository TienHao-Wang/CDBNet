from dinov3.models.vision_transformer import DinoVisionTransformer
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    image_path = r"E:\DINOv3 with CIP\test1.tif"
    image = Image.open(image_path).convert('RGB')
    W, H = image.size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = transform(image).unsqueeze(0).cuda()

    model = DinoVisionTransformer(img_size=512)
    model = model.cuda()
    x = model(tensor_image)
    print(x)
