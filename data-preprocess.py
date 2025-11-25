import argparse
import json
import shutil
import os
import random
from helpers import get_run_identification, unpack_dataset
import valohai
import cv2

random.seed(10)


def save_dataset(out_path, subset='train'):
    out_path = out_path[:-1]
    project_name, exec_id = get_run_identification()

    metadata = {
        'valohai.dataset-versions': [
            {
                'uri': f'dataset://demo-ships-aerial/{project_name}_{subset}_{exec_id}',
                'targeting_aliases': [f'dev_{subset}'],
                'valohai.tags': ['dev', 'ships-aerial'],
            },
        ],
    }

    for folder in os.listdir(out_path + '/' + subset):

        for file in os.listdir(out_path + '/' + subset + '/' + folder):
            metadata_path = os.path.join(out_path, subset, folder, f'{file}.metadata.json')
            with open(metadata_path, 'w') as outfile:
                json.dump(metadata, outfile)


def prepare_data(data_path, save_path, train_size, valid_size, test_size, image_size):
    sets = {'train': train_size, 'valid': valid_size, 'test': test_size}
    data_path = os.path.join(data_path, os.listdir(data_path)[0])

    for subset, size in sets.items():

        img_files_path = os.path.join(data_path, f'{subset}/images')
        lbl_files_path = os.path.join(data_path, f'{subset}/labels')

        img_destination_path = valohai.outputs(f'prep_dataset/{subset}/images')
        lbl_destination_path = valohai.outputs(f'prep_dataset/{subset}/labels')

        image_files = os.listdir(img_files_path)
        random_images = random.sample(image_files, size)

        for i, image_file in enumerate(random_images):
            image_path = os.path.join(img_files_path, image_file)
            image = cv2.imread(image_path)
            image = preprocess(image, image_size)
            cv2.imwrite(img_destination_path.path(image_file), image)

            shutil.move(lbl_files_path + f'/{image_file[:-3]}txt', lbl_destination_path.path('.'))

        save_dataset(save_path, subset)


def preprocess(img, img_size):
    img = cv2.resize(img, (img_size, img_size))

    # Possible preprocess - We proceed without since ultralytics manages that under the hood.
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img / 255.
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset parameters for ship aerial images")
    parser.add_argument('--train_size', type=int, default=20,
                        help='Number of samples in the training set')
    parser.add_argument('--valid_size', type=int, default=5,
                        help='Number of samples in the validation set')
    parser.add_argument('--test_size', type=int, default=5,
                        help='Number of samples in the test set')
    parser.add_argument('--image_size', type=int, default=768,
                        help='Size of the images')
    args = parser.parse_args()

    dataset_packed = valohai.inputs('dataset').path(process_archives=False)
    print('dataset_packed ', dataset_packed)

    output_path = valohai.outputs('prep_dataset').path('.')
    unpacked_dataset_path = unpack_dataset(dataset_packed, 'unpacked_dataset')

    prepare_data(unpacked_dataset_path, output_path, args.train_size, args.valid_size, args.test_size, args.image_size)
