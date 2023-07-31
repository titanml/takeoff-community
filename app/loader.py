"""HuggingFace model loader."""
# ───────────────────────────────────────────────────── imports ────────────────────────────────────────────────────── #

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from huggingface_hub import snapshot_download
import os
from tinydb import TinyDB, Query

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
#                                            HuggingFace Model Loader Class                                            #
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #


class HFModelLoader:
    """HuggingFace model loader."""

    def __init__(self, model_name_or_path):
        """Initialize the model loader.

        Args:
            model_name_or_path (str): model name or path
        """
        self.model_name_or_path = model_name_or_path

        self.local_save_file = os.path.join("/code", "models", model_name_or_path)
        self.exists_locally = os.path.exists(self.local_save_file)

        if self.exists_locally:
            self.model_type = self.detect_model_type(self.local_save_file)
        else:
            self.model_type = self.detect_model_type(self.model_name_or_path)

    def detect_model_type(self, model_name_or_path):
        """Detect the model type.

        Args:
            model_name_or_path (str): model name or path

        Returns:
            str: type of model
        """
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        architecture = config.architectures[0]

        if "conditionalgeneration" in architecture.lower():
            return "SEQ2SEQ"

        elif "causallm" in architecture.lower():
            return "CAUSAL"

        # for mosaic
        elif "gpt" in architecture.lower():
            return "CAUSAL"

    @property
    def automodel(self):
        if self.model_type == "SEQ2SEQ":
            loader = AutoModelForSeq2SeqLM

        elif self.model_type == "CAUSAL":
            loader = AutoModelForCausalLM

        else:
            raise ValueError(f"Unknown model type {self.model_type}")

        return loader

    def pytorch_model(self, model_name_or_path, device="cpu"):
        """Load a model from HuggingFace and return the loaded model and tokenizer"""
        model = self.automodel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return model, tokenizer

    def download_model(self, model_name_or_path, save_dir):
        """Download a model from HuggingFace and save it to disk.

        Args:
            model_name_or_path (str): model name or path
            save_dir (str): saved directory
        """
        # model name

        model, tokenizer = self.pytorch_model(model_name_or_path)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    def check_cache(self, model_name_or_path, cache_loc):
        cache = TinyDB(cache_loc)
        Model = Query()
        result = cache.search(Model.hf_name == model_name_or_path)

        if len(result) == 0:
            return model_name_or_path

        else:
            return f"TitanML/{result[0]['saved_name']}"

    def download_repo(self, model_name_or_path, save_dir, safe=False):
        """Download an entire repo from Huggingface, to avoid loading the model into RAM.

        Args:
            model_name_or_path (str): model name or path
            save_dir (str): saved directory
        """

        if safe:
            snapshot_patterns = ["*.msgpack", "*.h5", "*.bin", "coreml/**/*"]
        else:
            snapshot_patterns = ["*.msgpack", "*.h5", "*.safetensors", "coreml/**/*"]
        model_name_or_path = self.check_cache(model_name_or_path, "/code/app/model_cache_mapping.json")

        print(f"Model already converted. Using checkpoint at TitanML/{model_name_or_path}")
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=save_dir,
            local_dir_use_symlinks=False,  # force the load to be in the volume mount, not in cache.
            ignore_patterns=snapshot_patterns,
        )
