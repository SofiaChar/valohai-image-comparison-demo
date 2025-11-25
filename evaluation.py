from ultralytics import YOLO
import valohai
import shutil
import os


def save_to_outputs():
    path = 'runs/detect/val'
    save_path = valohai.outputs('detections')
    print('save_path ', save_path)
    for f in os.listdir(path):
        src = os.path.join(path, f)
        dst = save_path.path(f)

        shutil.copy(src, dst)


def evaluate_yolo():
    # Load a model
    model_path = valohai.inputs('model').path()

    model = YOLO(model_path)  # load our model

    # Validate the model
    metrics = model.val(data='/valohai/inputs/data_yaml/data.yaml', split='test', plots=True)
    print('map50-95', metrics.box.map)  # map50-95
    print('map50', metrics.box.map50)  # map50
    print('map75', metrics.box.map75)  # map75
    print('List map50-95 of each category: ', metrics.box.maps)  # a list contains map50-95 of each category

    save_to_outputs()


if __name__ == '__main__':
    evaluate_yolo()
