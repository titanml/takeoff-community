"""FastAPI main module."""
# ───────────────────────────────────────────────────── imports ────────────────────────────────────────────────────── #

import os
import subprocess

import ctranslate2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder

from .loader import HFModelLoader
from fastapi.staticfiles import StaticFiles
from .config import settings
from .utils import get_logger

logger = get_logger("main")

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
#                                                  FastAPI Main Class                                                  #
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

app = FastAPI(
    title="Titan Fabulinus API Demo",
)
app.mount("/demos", StaticFiles(directory="./app/static", html=True), name="demo")


def requires_safe(model_name_or_path: str) -> bool:
    safe_keys = [
        "Llama-2-7b-hf",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-13b-chat-hf",
    ]
    if any(key in model_name_or_path for key in safe_keys):
        return True
    else:
        return False


def ct_preprocess(model_name_or_path, model_path):
    # if model name or path is meta-llama/Llama-2-7b, then delete the added_tokens.json file in the model_name_or_path
    process_keys = [
        "Llama-2-7b-hf",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-13b-chat-hf",
    ]
    if any(key in model_name_or_path for key in process_keys):
        logger.info(f"Modifying vocab for {model_name_or_path}")
        added_tokens_file = os.path.join(model_path, "added_tokens.json")
        if os.path.exists(added_tokens_file):
            os.remove(added_tokens_file)
        else:
            logger.debug(f"Already removed {added_tokens_file}")


@app.on_event("startup")
async def startup():
    """Startup event.

    This function is called when the server starts up.
    - It will check if the model is downloaded and download it if it is not
    - It will call the CT2 Converter to convert the model to CTranslate2 format
    - It will then load the generator and tokenizer
    """
    global generator, tokenizer, model_type

    # Get the model name from the environment variable
    model_name_or_path = settings.model_name
    device = settings.device
    access_token = settings.access_token
    quant_type = settings.quant_type
    logger.info("Taking off!")
    logger.info("-----------------------")
    logger.info("Configuration")
    logger.info("-----------------------")
    logger.info(f"Model Name: {model_name_or_path}")
    logger.info(f"Device: {device}")
    logger.debug(f"Quantization Type: {quant_type}")  # only print this at debug level
    logger.info("-----------------------")

    if access_token:
        HfFolder.save_token(access_token)
    if not model_name_or_path:
        raise ValueError("MODEL_NAME is not set")

    logger.info("Starting model loader...")
    loader = HFModelLoader(model_name_or_path=model_name_or_path)
    model_type = loader.model_type

    # Download the model if it does not exist
    if not loader.exists_locally:
        logger.info(
            f"Downloading model... {model_name_or_path} to {loader.local_save_file}. Safe: {requires_safe(model_name_or_path)}"
        )
        loader.download_repo(
            model_name_or_path,
            loader.local_save_file,
            safe=requires_safe(model_name_or_path),
        )

    if model_type == "CAUSAL":
        model = ctranslate2.Generator
    elif model_type == "SEQ2SEQ":
        model = ctranslate2.Translator
    logger.info(f"Optimizing {model_type} model: " + loader.local_save_file)

    target_save_file = os.path.join(
        "/code/models", loader.local_save_file, "ct_output_models"
    )
    # Run the first command

    if not os.path.exists(target_save_file):
        ct_preprocess(model_name_or_path, loader.local_save_file)
        logger.info("Running CT2 Converter...")
        command1 = [
            "ct2-transformers-converter",
            "--model",
            loader.local_save_file,
            "--quantization",
            quant_type,
            "--output_dir",
            target_save_file,
            "--trust_remote_code",
            "--low_cpu_mem_usage",
        ]
        result_ct = subprocess.run(command1)

        # Check the exit status of the first command
        if result_ct.returncode == 0:
            logger.info("CT2 Converter finished successfully")
        else:
            # First command failed, so do not run the second command
            logger.info("CT2 Converter failed. Exiting...")
            raise SystemExit(1)
    else:
        logger.info("Found cached CT2 model. Skipping conversion...")

    logger.info("Loading generator...")
    generator = model(target_save_file, device=device)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        loader.local_save_file, trust_remote_code=True
    )

    logger.info("Optimization complete.")


class TextIn(BaseModel):
    """Pydantic model for prompt text."""

    text: str
    generate_max_length: int = 128
    sampling_topk: int = 1
    sampling_topp: float = 1.0
    sampling_temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


async def generate_stream_helper(
    text,
    generate_max_length,
    sampling_topk,
    sampling_topp,
    sampling_temperature,
    repetition_penalty,
    no_repeat_ngram_size,
):
    # sourcery skip: use-fstring-for-concatenation
    """Generate text from prompt.

    Args:
        text (str): prompt text

    Yields:
        str: generated text
    """
    prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    params = {
        "sampling_topk": sampling_topk,
        "sampling_topp": sampling_topp,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "sampling_temperature": sampling_temperature,
        "end_token": [tokenizer.eos_token_id],
    }
    # add the max_length parameter only for model 'CAUSAL', remove it for 'SEQ2SEQ'
    if model_type == "CAUSAL":
        params["max_length"] = generate_max_length

    elif model_type == "SEQ2SEQ":
        params["max_decoding_length"] = generate_max_length

    step_results = generator.generate_tokens(prompt_tokens, **params)

    for step_result in step_results:
        # this is for llama tokenizer to decode with space
        if step_result.token.startswith("▁"):
            yield " " + tokenizer.decode(
                step_result.token_id
            )  # pre pend a whitie space
        else:
            yield tokenizer.decode(step_result.token_id)


def generate_helper(
    text,
    generate_max_length,
    sampling_topk,
    sampling_topp,
    sampling_temperature,
    repetition_penalty,
    no_repeat_ngram_size,
):
    # sourcery skip: use-fstring-for-concatenation
    """Generate text from prompt.

    Args:
        text (str): prompt text

    Yields:
        str: generated text
    """
    prompt_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    params = {
        "sampling_topk": sampling_topk,
        "sampling_topp": sampling_topp,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "sampling_temperature": sampling_temperature,
        "end_token": [tokenizer.eos_token_id],
    }
    # add the max_length parameter only for model 'CAUSAL', remove it for 'SEQ2SEQ'
    if model_type == "CAUSAL":
        params["max_length"] = generate_max_length

    elif model_type == "SEQ2SEQ":
        params["max_decoding_length"] = generate_max_length

    step_results = generator.generate_tokens(prompt_tokens, **params)

    result = ""
    for step_result in step_results:
        # this is for llama tokenizer to decode with space
        if step_result.token.startswith("▁"):
            result += " " + tokenizer.decode(
                step_result.token_id
            )  # pre pend a whitie space
        else:
            result += tokenizer.decode(step_result.token_id)
    return result


@app.post("/generate_stream")
async def generate_stream(payload: TextIn):
    """Post request to generate text from prompt.

    Args:
        payload (TextIn): prompt text in pydantic model

    Returns:
        StreamingResponse: StreamingResponse of generated text
    """
    logger.debug(f"Prompt: {payload.text}")
    return StreamingResponse(
        generate_stream_helper(
            payload.text,
            payload.generate_max_length,
            payload.sampling_topk,
            payload.sampling_topp,
            payload.sampling_temperature,
            payload.repetition_penalty,
            payload.no_repeat_ngram_size,
        )
    )


@app.post("/generate")
async def generate(payload: TextIn):
    """Post request to generate text from prompt.

    Args:
        payload (TextIn): prompt text in pydantic model

    Returns:
        JSONResponse: JSONResponse of generated text
    """
    logger.debug(f"Prompt: {payload.text}")

    result_str = generate_helper(
        payload.text,
        payload.generate_max_length,
        payload.sampling_topk,
        payload.sampling_topp,
        payload.sampling_temperature,
        payload.repetition_penalty,
        payload.no_repeat_ngram_size,
    )

    return {"status": "success", "message": result_str}
