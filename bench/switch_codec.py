"""The `switch` codec: the library's when zarr-python has it, else the plugin copy."""

try:
    from zarr.codecs.switch import SwitchCodec, rle_cases
except ImportError:  # released zarr-python
    from switch_codec_plugin import SwitchCodec, rle_cases  # noqa: F401
