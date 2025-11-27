import argparse
import json
from ultralytics import YOLO


def create_data_yaml():
    import yaml
    save_path = '/valohai/outputs/data.yaml'
    data_yaml = {
        'train': '/valohai/inputs/train',
        'val': '/valohai/inputs/valid',
        'test': '/valohai/inputs/test',
        'nc': 1,
        'names': ['ship']
    }
    with open(save_path, 'w') as file:
        yaml.dump(data_yaml, file)

    metadata_path = f'{save_path}.metadata.json'
    print('metadata_path ', metadata_path)
    with open(metadata_path, 'w') as outfile:
        json.dump({"valohai.alias": 'data_yaml'}, outfile)
    return save_path


def save_model_alias():
    """
    YOLO saves the model weights to /valohai/outputs/train/weights,
    We want to save the metadata for the best.pt to create new version of Valohai Model
    """
    metadata = {
            "valohai.model-versions": ["model://ship-detection/"],  # creates new version of the model
    }

    metadata_path = "/valohai/outputs/train/weights/best.pt.metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def train_yolo(yolo_name, data_yaml, image_size, epochs, seed, batch_size, project):
    model = YOLO(yolo_name)

    # Training the model
    model.train(data=data_yaml,
                epochs=epochs,
                imgsz=image_size,
                seed=seed,
                batch=batch_size,
                project=project)

    save_model_alias(project)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training parameters for ship aerial images")
    parser.add_argument('--yolo_model_name', type=str, default='yolov8x.pt',
                        help='Model name')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')

    parser.add_argument('--image_size', type=int, default=768,
                        help='Size of the images')

    parser.add_argument('--optimizer', type=str, default='SGD')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--project', type=str, default='/valohai/outputs',
                        help='Save path for the training logs')
    args = parser.parse_args()

    yaml_path = create_data_yaml()
    train_yolo(args.yolo_model_name, yaml_path, args.image_size, args.epochs, args.seed, args.batch_size, args.project)
