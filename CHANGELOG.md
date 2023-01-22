# Changelog

## [v2] - 2022-21-16

- Added 3 more descent algorithms: SGD+Momentum, RMSProp, AdamOptimizer
- ```display``` parameter to toggle display output during training loop

## [v1] - 2023-01-16

### Added

- DataLoader, changed how training data is loaded into model
- Customize batch sizes and enable shuffling setting for data loading, rather than single label/feature pair.
- Added Stochastic Gradient Descent, also abstracted learning rate term for customizable descent algorithms.
- ```timed``` parameter for training loop to check current time at each epoch stamp.

### Changed

- Up-to-date examples for Iris & MNIST.

### Removed

- Old train/test implementation, including iris and mnist examples