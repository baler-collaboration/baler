# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

from baler.modules import helper
import time
from datetime import datetime


def test_create_new_project():
    workspace_name = "test_workspace"
    project_name = "test_project"
    base_path = os.path.join("test_workspaces")
    workspace_path = os.path.join(base_path, workspace_name)
    project_path = os.path.join(workspace_path, project_name)

    # Ensure the project doesn't exist yet
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)

    helper.create_new_project(workspace_name, project_name, base_path=base_path)

    # Verify that the project was created successfully
    assert os.path.exists(workspace_path)
    assert os.path.exists(project_path)

    required_directories = [
        os.path.join(workspace_path, "data"),
        os.path.join(project_path, "config"),
        os.path.join(project_path, "output", "compressed_output"),
        os.path.join(project_path, "output", "decompressed_output"),
        os.path.join(project_path, "output", "plotting"),
        os.path.join(project_path, "output", "training"),
    ]
    for directory in required_directories:
        print(directory)
        assert os.path.exists(directory)

    # Clean up after the test
    shutil.rmtree(base_path)


def test_green_code_tracking_no_verbose(capsys):
    log_file = "green_code_tracking_test.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        start_time = time.time()
        time.sleep(0.01)
        end_time = time.time()
        title = "Test Process No Verbose"

        helper.green_code_tracking(
            start_time, end_time, title, verbose=False, testing=True
        )

        # Check file content
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert title in content
            assert f"Total time taken: {end_time - start_time:.3f} seconds" in content

        # Check that nothing was printed to stdout
        captured = capsys.readouterr()
        assert captured.out == ""

    finally:
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)


def test_green_code_tracking_with_verbose(capsys):
    log_file = "green_code_tracking_test.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        start_time = time.time()
        time.sleep(0.01)
        end_time = time.time()
        title = "Test Process Verbose"

        helper.green_code_tracking(
            start_time, end_time, title, verbose=True, testing=True
        )

        # Check file content
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            content = f.read()
            assert title in content
            assert f"Total time taken: {end_time - start_time:.3f} seconds" in content

        # Check stdout
        captured = capsys.readouterr()
        assert "GREEN CODE INITIATIVE" in captured.out
        assert (
            f"Total time taken for {title}: {end_time - start_time:.3f} seconds"
            in captured.out
        )
        assert f"{title} complete." in captured.out

    finally:
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)


def test_green_code_tracking_appends_to_file():
    log_file = "green_code_tracking_test.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    try:
        # First call
        start_time1 = time.time()
        time.sleep(0.01)
        end_time1 = time.time()
        title1 = "PYTEST - First Process"
        helper.green_code_tracking(start_time1, end_time1, title1, testing=True)

        # Second call
        start_time2 = time.time()
        time.sleep(0.01)
        end_time2 = time.time()
        title2 = "PYTEST - Second Process"
        helper.green_code_tracking(start_time2, end_time2, title2, testing=True)

        # Check file content
        assert os.path.exists(log_file)
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert title1 in lines[0]
            assert f"{end_time1 - start_time1:.3f}" in lines[0]
            assert title2 in lines[1]
            assert f"{end_time2 - start_time2:.3f}" in lines[1]

    finally:
        # Cleanup
        if os.path.exists(log_file):
            os.remove(log_file)
