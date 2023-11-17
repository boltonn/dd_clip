from enum import StrEnum


class Ctranslate2ComputeType(StrEnum):
    """Compute type to use for the model.

    Notes:
        default: keep the same quantization that was used during model conversion (see Implicit type conversion on load for exceptions)
        auto: use the fastest computation type that is supported on this system and device
    """

    default = "default"
    auto = "auto"
    int8 = "int8"
    int8_float32 = "int8_float32"
    int8_float16 = "int8_float16"
    int8_bfloat16 = "int8_bfloat16"
    int16 = "int16"
    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"
