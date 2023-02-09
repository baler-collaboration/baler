import os
import shutil

from baler.modules import helper


def test_create_new_project():
    project_name = "test_project"
    base_path = "./tmp"
    project_path = os.path.join(base_path, project_name)

    # Ensure the project doesn't exist yet
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

    helper.create_new_project(project_name, base_path)

    # Verify that the project was created successfully
    assert os.path.exists(project_path)
    for directory in [
        "compressed_output",
        "decompressed_output",
        "plotting",
        "training",
        "model",
    ]:
        assert os.path.exists(os.path.join(project_path, directory))

    # Clean up after the test
    shutil.rmtree(base_path)
