from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Loads the Qwen model and generates answers from context."""
    MODEL = "Qwen/Qwen3-0.6B"

    def __init__(self) -> None:
        """Initialize the model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        dtype = "auto" if self.device == "cuda" else torch.float32
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.MODEL, dtype=dtype, device_map=self.device
        )
        self.model.eval()

    def answer(self, question: str, context: str, max_new: int = 256) -> str:
        """Generate the answer from context."""
        system = (
            "You are a vLLM documentation assistant. Answer the question "
            "using only the retrieved context. The first passages are the "
            "most relevant, ignore irrelevant ones. Start with the direct "
            "answer, then one or two sentences of explanation. Only name "
            "commands, flags and values that appear in the context. If the "
            "context lists several relevant items, include them all."
            "If the context does not answer the question, say so."
        )
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # torch.inference_mode() skips autograd and saves memory
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]  # drop the echoed prompt
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return str(text).strip()
