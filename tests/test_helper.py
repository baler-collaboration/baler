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

import numpy as np
import shutil

from baler.modules import helper


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


def test_convert_to_original_types_rounds_integer_columns():
    data = np.array(
        [
            [2.999, 0.4, 1.25],
            [3.001, 0.6, -1.75],
            [-0.6, 1.5, 0.5],
        ]
    )
    out = helper.convert_to_original_types(data.copy(), ["int", "int", "float64"])
    # Integer columns are rounded, not truncated (astype(int) would give 2, 3, 0
    # and 0, 0, 1)
    np.testing.assert_array_equal(out[:, 0], [3, 3, -1])
    np.testing.assert_array_equal(out[:, 1], [0, 1, 2])
    # Float columns are unchanged
    np.testing.assert_array_equal(out[:, 2], data[:, 2])
