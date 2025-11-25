import json
import logging
import time

logger = logging.getLogger(__name__)


def get_run_identification():
    try:
        with open('/valohai/config/execution.json') as f:
            exec_details = json.load(f)
        project_name = exec_details['valohai.project-name'].split('/')[1]
        exec_id = exec_details['valohai.execution-id']
    except FileNotFoundError:
        project_name = 'test'
        exec_id = str(int(time.time()))
    return project_name, exec_id

def unpack_dataset(dataset_path, output_to='./unpacked_dataset'):
    import zipfile
    with zipfile.ZipFile(f"{dataset_path}", "r") as zip_ref:
        zip_ref.extractall(output_to)

    return output_to
