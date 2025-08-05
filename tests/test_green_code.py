import os
import csv
import time
from unittest.mock import MagicMock, patch
import pytest
from baler.modules.green_code import GreenCodeTracker

@pytest.fixture
def mock_specs():
    """A fixture to provide a consistent, fake set of system specs."""
    return {
        "CPUmodel": "Fake CPU v1",
        "Number of CPU cores": 8,
        "GPU model": "Fake GPU v1",
        "Number of GPU cores": 1024,
        "Memory available (GB)": 16.0,
    }

class TestGreenCodeTracker:
    """Group of tests for the GreenCodeTracker class."""

    def test_init_creates_file_and_header(self, tmp_path, mock_specs, mocker):
        """
        Test that the constructor creates a CSV file with the correct header.
        We mock _get_system_specs to isolate the test from the hardware.
        """
        mocker.patch.object(GreenCodeTracker, '_get_system_specs', return_value=mock_specs)
        test_file = tmp_path / "green_code_tracking_test.csv"
        tracker = GreenCodeTracker(file_path=str(test_file))
        

        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == tracker.headers

    def test_track_method_writes_correct_row(self, tmp_path, mock_specs, mocker):
        """Test that the track() method appends a correctly formatted row."""
        mocker.patch.object(GreenCodeTracker, '_get_system_specs', return_value=mock_specs)
        test_file = tmp_path / "green_code_tracking_test.csv"
        tracker = GreenCodeTracker(file_path=str(test_file))

        tracker.track(start=100.0, end=102.5, title="Test Task")

        with open(tracker.file_name, 'r') as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            row = reader[0]
            assert row["Title"] == "Test Task"
            assert row["Runtime(format HH:MM:SS)"] == "00:00:02"
            assert row["CPUmodel"] == "Fake CPU v1"
            assert row["GPU model"] == "Fake GPU v1"

    def test_time_context_manager(self, tmp_path, mock_specs, mocker):
        """Test that the time() context manager works as expected."""
        mocker.patch.object(GreenCodeTracker, '_get_system_specs', return_value=mock_specs)
        test_file = tmp_path / "green_code_tracking_test.csv"
        tracker = GreenCodeTracker(file_path=str(test_file))

        with tracker.time("Context Task"):
            time.sleep(0.01)

        with open(tracker.file_name, 'r') as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert reader[0]["Title"] == "Context Task"
            assert reader[0]["Runtime(format HH:MM:SS)"] == "00:00:00"

    def test_get_system_specs_with_gpu(self, tmp_path, mocker):
        """Test spec gathering on a mocked system WITH an NVIDIA GPU."""
        # Mock all external calls
        mocker.patch('cpuinfo.get_cpu_info', return_value={'brand_raw': 'Mocked Intel CPU'})
        mocker.patch('psutil.cpu_count', return_value=12)
        mocker.patch('psutil.virtual_memory', return_value=MagicMock(total=32 * 1024**3))
        
        # Mock pynvml to simulate a detected GPU
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_pynvml.nvmlDeviceGetName.return_value = "Mocked NVIDIA GPU"
        mock_pynvml.nvmlDeviceGetMultiProcessorCount.return_value = 40
        mocker.patch('baler.modules.green_code.pynvml', mock_pynvml, create=True)
        mocker.patch('baler.modules.green_code.NVIDIA_SMI_AVAILABLE', True)
                
        test_file = tmp_path / "green_code_tracking_test.csv"
        tracker = GreenCodeTracker(file_path=str(test_file))
        specs = tracker._get_system_specs()
        
        assert specs["CPUmodel"] == "Mocked Intel CPU"
        assert specs["Number of CPU cores"] == 12
        assert specs["GPU model"] == "Mocked NVIDIA GPU"
        assert specs["Number of GPU cores"] == 40
        assert specs["Memory available (GB)"] == 32.0

    def test_get_system_specs_without_gpu(self, tmp_path, mocker):
        """Test spec gathering on a mocked system WITHOUT an NVIDIA GPU."""
        mocker.patch('cpuinfo.get_cpu_info', return_value={'brand_raw': 'Mocked AMD CPU'})
        mocker.patch('psutil.cpu_count', return_value=4)
        mocker.patch('psutil.virtual_memory', return_value=MagicMock(total=8 * 1024**3))
        
        # Simulate pynvml not being available
        mocker.patch('baler.modules.green_code.NVIDIA_SMI_AVAILABLE', False)

        test_file = tmp_path / "green_code_tracking_test.csv"
        tracker = GreenCodeTracker(file_path=str(test_file))
        specs = tracker._get_system_specs()

        assert specs["GPU model"] == "N/A"
        assert specs["Number of GPU cores"] == "N/A"
        assert specs["CPUmodel"] == "Mocked AMD CPU"