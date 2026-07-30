# Legacy Data Module

**⚠️ LEGACY - DEPRECATED - DO NOT EXTEND**

This directory contains legacy data loading code for backward compatibility only.

## Status

- **Deprecated**: This module is deprecated and should not be used for new development.
- **Do Not Extend**: Do not implement new functionality in this directory.
- **Migration**: All new data pipeline development must occur in the `data/` directory at the project root.

## Migration Path

New data pipeline implementation is located at:

```
data/
├── dataset.py
├── inspection.py
├── statistics.py
├── transforms.py
└── dataloader.py
```

## Backward Compatibility

The existing `load_cifar10.py` is preserved for backward compatibility with existing code that depends on it. However, all new code should import from the `data/` package instead.

## Future

This directory will be removed once all dependent code has been migrated to the new data pipeline.
