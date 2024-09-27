from data_preprocessing import load_data
from models.efficientnet_model import build_efficientnet_model
from models.resnet_model import build_resnet_model
from models.vit_model import build_vit_model
from models.cvt_model import build_cvt_model
from models.swin_model import build_swin_model
from training import train_model
from utils import show_image_samples

def main(model_type='efficientnet'):
    # Load data
    train_gen, valid_gen, target_dict = load_data()

    # Show sample images
    show_image_samples(train_gen)

    # Build and train the model
    if model_type == 'model/efficientnet':
        model = build_efficientnet_model(num_classes=len(target_dict))
    elif model_type == 'model/resnet':
        model = build_resnet_model(num_classes=len(target_dict))
    elif model_type == 'model/vit':
        model = build_vit_model(num_classes=len(target_dict))
    elif model_type == 'model/cvt':
        model = build_cvt_model(num_classes=len(target_dict))
    elif model_type == 'model/swin':
        model = build_swin_model(num_classes=len(target_dict))
    else:
        raise ValueError("Invalid model type specified!")

    train_model(model, train_gen, valid_gen)

if __name__ == "__main__":
    main(model_type='efficientnet')  # Change to 'resnet', 'vit', 'cvt', or 'swin' as needed
