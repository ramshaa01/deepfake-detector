import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import timm

def test_model_loads_and_infers():
    checkpoint_path = "model/checkpoints/day32_finetuned_converged.pth"
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    
    # Load model using TIMM (which was used during training)
    model = timm.create_model("efficientnet_b0", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 1)
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Test on a few sample images
    sample_images = [
        "results/gradcam_correct_samples/fake_fake_034KKELVWP.jpg",
        "results/gradcam_correct_samples/real_real_13349.jpg"
    ]
    
    for img_path in sample_images:
        assert os.path.exists(img_path), f"Sample image missing: {img_path}"
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(tensor)
            
        prob = torch.sigmoid(output).item()
        assert 0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}"
