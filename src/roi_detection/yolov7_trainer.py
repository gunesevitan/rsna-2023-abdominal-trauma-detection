import sys
import argparse

sys.path.append('..')
import settings
sys.path.append('../../venv/lib/python3.11/site-packages')
from yolov7.det import train, val


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = settings.MODELS / args.model_directory

    if args.mode == 'training':

        train.run(
            data=model_directory / 'dataset.yaml',
            imgsz=512,
            batch=16,
            weights=model_directory / 'yolov7-tiny.pt',
            cfg=model_directory / 'yolov7-tiny.yaml',
            epochs=2,
            project=model_directory,
            name='experiment',
            exist_ok=False,
            hyp=model_directory / 'hyp.yaml',
            device='0',
            optimizer='AdamW'
        )

    elif args.mode == 'inference':

        val.run(
            data=model_directory / 'dataset.yaml',
            weights=model_directory / 'experiment' / 'weights' / 'best.pt',
            batch_size=32,
            imgsz=512,
            conf_thres=0.001,
            iou_thres=0.6,
            task='val',
            device='',
            workers=8,
            single_cls=False,
            augment=False,
            verbose=True,
            save_txt=False,
            save_hybrid=False,
            save_conf=False,
            save_json=False,
            project=model_directory,
            name='experiment',
            exist_ok=False,
            half=True,
            dnn=False,
            plots=True
        )

    else:
        raise ValueError(f'Invalid mode {args.mode}')
