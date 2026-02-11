# Changelog

## Unreleased

### Added
- Support for loading and operating external PyTorch `torch.utils.data.Dataset` objects (#382)
  - New `load_external_dataset` function in `data_processing.py`
  - Updated `perform_training` in `baler.py` to handle external datasets
  - Updated `train` function in `training.py` to accept DataLoaders directly
  - Added documentation and examples for using external datasets
  - Added unit tests for the new functionality

### Changed

### Fixed 