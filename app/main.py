"""FastAPI main module."""
# ───────────────────────────────────────────────────── imports ────────────────────────────────────────────────────── #

import os
import subprocess
from typing import List, Union

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

# batch imports
import uuid
import asyncio
import time
from threading import Thread

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
        "meta-llama/Llama-2-7b-hf",
        "meta_llama/Llama-2-7b-chat-hf",
        "meta_llama/Llama-2-13b-hf",
        "meta_llama/Llama-2-13b-chat-hf",
        "stabilityai/StableBeluga-7B",
    ]
    if any(key in model_name_or_path for key in safe_keys):
        return True
    else:
        return False


def ct_preprocess(model_name_or_path, model_path):
    # if model name or path is meta-llama/Llama-2-7b, then delete the added_tokens.json file in the model_name_or_path
    process_keys = ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "Llama-2-13b-hf", "Llama-2-13b-chat-hf"]
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

    # batch generation globals
    global generator_batched_fn, batched_input_queue, batched_output_queues, batched_results, executor, max_batch_size, future, batch_timeout, result_sleep_timeout

    # Get the model name from the environment variable
    model_name_or_path = settings.model_name
    device = settings.device
    access_token = settings.access_token
    quant_type = settings.quant_type
    max_batch_size = settings.max_batch_size
    disable_batching = settings.disable_batching
    batch_timeout = settings.batch_timeout / 1000.0  # ms to seconds
    result_sleep_timeout = settings.result_sleep_timeout / 1000.0  # ms to seconds

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
    loader = HFModelLoader(
        model_name_or_path=model_name_or_path,
    )
    model_type = loader.model_type

    # Download the model if it is not already downloaded
    # We need to check the cache first to get the model name
    loader.download_repo_or_skip(quant_type=quant_type, safe=requires_safe(model_name_or_path))

    if model_type == "CAUSAL":
        model = ctranslate2.Generator
    elif model_type == "SEQ2SEQ":
        model = ctranslate2.Translator

    new_model_quant_name = os.path.join("/code/models", loader.local_save_file, f"ct_output_models_{quant_type}")
    old_model_quant_name = os.path.join("/code/models", loader.local_save_file, f"ct_output_models")

    new_model_exists = os.path.exists(new_model_quant_name)
    old_model_exists = os.path.exists(old_model_quant_name)

    if new_model_exists:
        target_save_file = new_model_quant_name
        neither_exist = False

    elif old_model_exists:
        target_save_file = old_model_quant_name
        neither_exist = False
    else:
        # we must have skipped if this is the case
        # this means we have downloaded a enw model and need to convert it
        target_save_file = new_model_quant_name
        neither_exist = True

    if neither_exist:
        logger.info(f"Optimizing {model_type} model: " + loader.local_save_file)

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

    # The models saved in the cache are named differently than the ones that we now save with quantization
    # So we need to check if the quantized model exists, and if not, use the old model naming convention

    generator = model(target_save_file, device=device)

    if model_type == "CAUSAL":
        generator_batched_fn = generator.generate_batch

    elif model_type == "SEQ2SEQ":
        generator_batched_fn = generator.translate_batch

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(loader.local_save_file, trust_remote_code=True)

    logger.info("Optimization complete.")

    batched_input_queue = asyncio.Queue()

    # We keep multiple queues for output, each maintained by its own thread.
    batched_output_queues = {}  # <ThreadId, asyncio.Queue>
    batched_results = {}
    batched_generation_params = {}

    if not disable_batching:
        # # launch a second thread to process queued data for batching.
        # # if you use the streaming endpoint there might be some overhead from this thread
        # # so you can disable it by setting DISABLE_BATCHING to True

        ml_task = Thread(target=process_batched_data)
        ml_task.start()


class TextIn(BaseModel):
    """Pydantic model for prompt text."""

    text: str
    generate_max_length: int = 128
    sampling_topk: int = 1
    sampling_topp: float = 1.0
    sampling_temperature: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


class BatchedTextIn(BaseModel):
    """Pydantic model for batched text prompts"""

    text: Union[List[str], str]
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

    unwanted_patterns = [tokenizer.eos_token, "User", "Assistant"]
    for step_result in step_results:
        # Decode the token content
        if step_result.token.startswith("▁"):
            token_content = " " + tokenizer.decode(step_result.token_id)
        else:
            token_content = tokenizer.decode(step_result.token_id)
        if token_content in unwanted_patterns:
            break
        else:
            for char in token_content:
                yield char
                await asyncio.sleep(0)


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


def process_batched_data():
    # This function runs in a separate thread
    # It listens to the batched_input_queue and pops out up to a size of max_batch_size.
    # Then it runs ct2 batched generation, and puts the results in batched_output_queue

    # this thread is always running, and always listening to the queue
    while True:
        # do nothing if the queue is empty
        if batched_input_queue.empty():
            continue

        # Read Input Queue
        # ================

        # collect up to max_batch_size items from the queue
        batched_input = []
        batch_ids = []
        thread_ids = []
        current_queue_size = batched_input_queue.qsize()

        if current_queue_size < max_batch_size:
            time.sleep(batch_timeout)
            # sleep for 50 ms to wait for batched inputs
            # if we dont do this then there is a mini-race condition where
            # the ML thread immediately plucks the first results from the
            # input queue as it arrives and does low batch inference. If the
            # server thread is busy trying to push to the queue then a 50ms gap will
            # let it populate the input queue, and if we just have low batches
            # being requested then we just wait a bit.

        post_wait_current_queue_size = batched_input_queue.qsize()
        num_batched = 0
        item_considered = 0
        while (num_batched < max_batch_size) and (item_considered < post_wait_current_queue_size):
            queue_item = batched_input_queue.get_nowait()  # QueueItem(id, text, generation_params)

            # Generation Parameter Batching Behaviour
            # =======================================
            # We only batch items that have the same generation parameters
            # This is because we need a consistent set of generation parameters for a single
            # batch of requests.
            # We do this by looking at the parameters of the first item in the queue, and using
            # this as a target set of parameters and we look for other items in the queue that
            # share those parameters.

            # We continue until either:
            # 1. We have found max_batch_size items with the same generation parameters
            # 2. We have considered all items in the queue (current_queue_size)

            if item_considered == 0:
                # if this is the first item we have considered, then we set the target generation params
                target_batched_generation_params = queue_item.generation_params
            else:
                # if the current item has different generation parameters, then we skip it
                # we increment item_considered so that we know we have looked at it
                candidate_batched_generation_params = queue_item.generation_params

                if candidate_batched_generation_params != target_batched_generation_params:
                    item_considered += 1
                    continue

            # if we get here, then we have found an item with the same generation parameters
            # we add it to the batch and increment the counters
            batch_ids.append(queue_item.id)
            batched_input.append(queue_item.text)
            thread_ids.append(queue_item.thread_id)
            num_batched += 1
            item_considered += 1

        logger.debug(f"Processing with batch size {num_batched} from a queue of size {post_wait_current_queue_size}")
        # Generation Logic

        # Pre Process Inputs
        # ==================
        encoded_texts = tokenizer.batch_encode_plus(batched_input)

        # cTranslate2 requires a list of tokens (not token ids, but sub-words instead) so we have to un-tokenize the batch
        # HF doesn't have a batched version of this function, so we have to do it row by row
        prompt_tokens = [tokenizer.convert_ids_to_tokens(encoded_text) for encoded_text in encoded_texts.input_ids]

        # Generation
        # ==========
        results = generator_batched_fn(prompt_tokens, **target_batched_generation_params)

        # Post Process Outputs
        # ====================
        if model_type == "CAUSAL":
            decoded_results = [tokenizer.decode(result.sequences_ids[0]) for result in results]

        elif model_type == "SEQ2SEQ":
            decoded_results = [tokenizer.convert_tokens_to_string(result.hypotheses[0]) for result in results]

        # Put Results in Output Queue
        # ===========================
        for result_text, result_id, result_thread_id in zip(decoded_results, batch_ids, thread_ids):
            # we put the result in each queue based on which thread it came from
            batched_output_queues[result_thread_id].put_nowait(
                QueueItem(
                    id=result_id,
                    thread_id=result_thread_id,
                    text=result_text,
                    generation_params=target_batched_generation_params,
                )
            )

        logger.debug(f"Successfully put {len(batch_ids)} results into the results queue")


# Class for items sitting in both the result and the input queue
class QueueItem(BaseModel):
    id: str  # str(uuid())
    thread_id: str  # str(uuid())
    text: str
    generation_params: dict


@app.post("/generate")
async def generate(payload: BatchedTextIn):
    """Post request to generate text from prompt.

    Args:
        payload (TextIn): prompt text in pydantic model

    Returns:
        JSONResponse: JSONResponse of generated text
    """
    thread_id = str(uuid.uuid4())
    batched_output_queues[thread_id] = asyncio.Queue()
    text = payload.text  # List[str]

    if isinstance(text, str):
        text = [text]

    # Set Up Generation Parameters
    # ============================
    # This is a bit awkward - different batches can be sent with different generation params
    # and there is no way to satisfy both.
    # We have to pick one - so each thread can update global params, and batches
    # waiting in the queue will use the latest ones.
    # Not sure if this opens up race conditions or not.

    # TODO Jamie:
    # We can batch by reading the queue looking for entries that have the same generation params
    # and then running the batched generation on those.
    # This will be slightly less efficient probably, but will allow users to transparantly
    # change generation params between requests.

    batched_generation_params = {
        "sampling_topk": payload.sampling_topk,
        "sampling_topp": payload.sampling_topp,
        "sampling_temperature": payload.sampling_temperature,
        "repetition_penalty": payload.repetition_penalty,
        "no_repeat_ngram_size": payload.no_repeat_ngram_size,
    }

    # add the max_length parameter only for model 'CAUSAL', remove it for 'SEQ2SEQ'
    if model_type == "CAUSAL":
        batched_generation_params["max_length"] = payload.generate_max_length
        batched_generation_params["include_prompt_in_result"] = False

    elif model_type == "SEQ2SEQ":
        batched_generation_params["max_decoding_length"] = payload.generate_max_length

    # Pushing Input Queue
    # ===================

    # Add Inputs to Input Queue
    ids_to_idx = {}
    all_ids = []
    logger.debug(f"Thread {thread_id} is adding {len(text)} items to queue")

    for batch_idx, batch_item in enumerate(text):
        item_id = str(uuid.uuid4())

        all_ids.append(item_id)
        ids_to_idx[item_id] = batch_idx

        await batched_input_queue.put(
            QueueItem(id=item_id, thread_id=thread_id, text=batch_item, generation_params=batched_generation_params)
        )

    # Reading Output Queue
    # ====================

    results = [None for _ in range(len(all_ids))]
    ready_results = 0

    while ready_results < len(all_ids):
        # Check if there are any results in the queue for this thread
        # ===========================================================
        result_queue_item = await batched_output_queues[thread_id].get()  # QueueItem
        result_id = result_queue_item.id
        result_text = result_queue_item.text
        results[ids_to_idx[result_id]] = result_text
        ready_results += 1

    logger.debug("Thread {} returned {} of {} items".format(thread_id, ready_results, len(all_ids)))

    if len(results) == 1:
        results = results[0]

    return {"status": "success", "message": results}
