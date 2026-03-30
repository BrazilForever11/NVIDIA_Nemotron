import csv
import os
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
#SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = Path.cwd()
DEFAULT_CSV_PATH = (SCRIPT_DIR / ".." / "data" / "train.csv").resolve()
DEFAULT_MODEL_DIR = (SCRIPT_DIR / ".." / "models" / DEFAULT_MODEL_ID.replace("/", "--")).resolve()
#DEFAULT_MODEL_DIR = (SCRIPT_DIR / "models" / DEFAULT_MODEL_ID.replace("/", "--")).resolve()


# Edit these values, then run each cell below in order.
MODEL_ID = DEFAULT_MODEL_ID
CSV_PATH = DEFAULT_CSV_PATH
PROMPT_COLUMN = "prompt"
NUM_EXAMPLES = 3
DOWNLOAD_DIR = DEFAULT_MODEL_DIR
# set HF_TOKEN env var or HUGGINGFACE_HUB_TOKEN env var to avoid download prompts

os.environ["HF_TOKEN"] =
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 128
DEVICE_MAP = "auto"
DTYPE = "auto"  # auto, bfloat16, float16, float32
DO_SAMPLE = False
TEMPERATURE = 0.7


def import_dependencies():
	try:
		import torch
		from huggingface_hub import snapshot_download
		from transformers import AutoModelForCausalLM, AutoTokenizer
	except ImportError as exc:
		raise SystemExit(
			"Missing dependencies. Install them with: pip install torch transformers huggingface_hub accelerate"
		) from exc

	return torch, snapshot_download, AutoModelForCausalLM, AutoTokenizer


def load_prompts(csv_path: Path, prompt_column: str, num_examples: int) -> list[str]:
	if num_examples <= 0:
		raise ValueError("NUM_EXAMPLES must be greater than 0.")

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	prompts: list[str] = []
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None or prompt_column not in reader.fieldnames:
			raise KeyError(
				f"Column '{prompt_column}' not found in {csv_path}. Available columns: {reader.fieldnames}"
			)

		for row in reader:
			value = (row.get(prompt_column) or "").strip()
			if not value:
				continue
			prompts.append(value)
			if len(prompts) >= num_examples:
				break

	if not prompts:
		raise ValueError(f"No non-empty prompts found in column '{prompt_column}' of {csv_path}.")

	return prompts


def download_model(snapshot_download, model_id: str, download_dir: Path, hf_token: str | None) -> Path:
	download_dir.mkdir(parents=True, exist_ok=True)
	local_path = snapshot_download(
		repo_id=model_id,
		local_dir=str(download_dir),
		token=hf_token,
		resume_download=True,
	)
	return Path(local_path).resolve()


def resolve_dtype(torch_module, dtype_name: str):
	if dtype_name == "auto":
		return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32
	return getattr(torch_module, dtype_name)


def resolve_input_device(torch_module, model) -> str:
	if torch_module.cuda.is_available():
		return "cuda"

	try:
		return str(next(model.parameters()).device)
	except StopIteration:
		return "cpu"


def build_generate_kwargs(max_new_tokens: int, do_sample: bool, temperature: float, tokenizer) -> dict:
	kwargs = {
		"max_new_tokens": max_new_tokens,
		"do_sample": do_sample,
		"pad_token_id": tokenizer.eos_token_id,
		"eos_token_id": tokenizer.eos_token_id,
	}
	if do_sample:
		kwargs["temperature"] = temperature
	return kwargs


def generate_completion(torch_module, model, tokenizer, prompt: str) -> str:
	tokenized = tokenizer(
		prompt,
		return_tensors="pt",
		truncation=True,
		max_length=MAX_INPUT_TOKENS,
	)
	tokenized = tokenized.to(resolve_input_device(torch_module, model))

	with torch_module.no_grad():
		outputs = model.generate(
			**tokenized,
			**build_generate_kwargs(MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE, tokenizer),
		)

	prompt_length = tokenized["input_ids"].shape[1]
	return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def show_examples(torch_module, model, tokenizer, prompts: list[str]) -> None:
	for index, prompt in enumerate(prompts, start=1):
		completion = generate_completion(torch_module, model, tokenizer, prompt)
		print(f"\n=== Example {index} ===")
		print("Prompt:")
		print(prompt)
		print("\nCompletion:")
		print(completion or "<empty completion>")



torch, snapshot_download, AutoModelForCausalLM, AutoTokenizer = import_dependencies()
torch_dtype = resolve_dtype(torch, DTYPE)

if not torch.cuda.is_available():
	print(
		"Warning: CUDA was not detected. Loading this 30B model on CPU is likely impractical.",
		file=sys.stderr,
		flush=True,
	)



print(f"Downloading or reusing snapshot for {MODEL_ID}...", flush=True)
local_model_dir = download_model(snapshot_download, MODEL_ID, DOWNLOAD_DIR.resolve(), HF_TOKEN)
print(f"Model files available at: {local_model_dir}", flush=True)



prompts = load_prompts(CSV_PATH.resolve(), PROMPT_COLUMN, NUM_EXAMPLES)
print(f"Loaded {len(prompts)} prompts from {CSV_PATH}")
prompts



print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
	local_model_dir,
	torch_dtype=torch_dtype,
	trust_remote_code=True,
	device_map=DEVICE_MAP,
)



prompt = prompts[0]
completion = generate_completion(torch, model, tokenizer, prompt)
print(prompt)
print("\nCompletion:\n")
print(completion or "<empty completion>")



show_examples(torch, model, tokenizer, prompts)
