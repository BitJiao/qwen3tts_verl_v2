"""Register Qwen3-TTS classes with Hugging Face auto classes.

Import this module through ``model.external_lib=recipe.qwen3_tts.register`` before
verl builds ``HFModelConfig``.
"""

from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor


def _safe_register(register_fn, *args, **kwargs):
    try:
        register_fn(*args, **kwargs)
    except ValueError as exc:
        message = str(exc).lower()
        if "already" not in message:
            raise


_safe_register(AutoConfig.register, "qwen3_tts", Qwen3TTSConfig)
_safe_register(AutoModel.register, Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
_safe_register(AutoProcessor.register, Qwen3TTSConfig, Qwen3TTSProcessor)
