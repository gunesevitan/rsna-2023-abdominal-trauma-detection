import sys
import argparse
import yaml
from ultralytics import YOLO

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_directory = settings.MODELS / args.model_directory
    hyp = yaml.load(open(model_directory / 'hyp.yaml', 'r'), Loader=yaml.FullLoader)

    model_path = model_directory / 'yolov8n.pt'
    model = YOLO(model_path)
    data = model_directory / 'dataset.yaml'

    results = model.train(
        data=data,
        epochs=100,
        patience=100,
        batch=16,
        imgsz=512,
        project=model_directory,
        name='experiment',
        exist_ok=False,
        optimizer='AdamW',
        seed=42,
        deterministic=False,
        **hyp
    )
