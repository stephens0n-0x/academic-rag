import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from app.core.config import config


def _get_device() -> str:
    """Resolve the optimal available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_llm() -> HuggingFacePipeline:
    """
    Initialize the language model and wrap it in a LangChain-compatible pipeline.
    Returns a HuggingFacePipeline instance ready for chain composition.
    """
    model_cfg = config["model"]
    device = model_cfg.get("device", _get_device())
    dtype = getattr(torch, model_cfg["torch_dtype"])

    print(f"[model] Loading {model_cfg['name']} on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    # HuggingFace pipeline wraps the model and tokenizer into a single callable
    # that LangChain chains can invoke directly without manual tokenization
    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=model_cfg["max_new_tokens"],
        temperature=model_cfg["temperature"],
        top_p=model_cfg["top_p"],
        repetition_penalty=model_cfg["repetition_penalty"],
        do_sample=True,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)