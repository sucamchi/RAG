from typing import Any


class Generator:
    """Loads the Qwen model and generates answers from context."""
    MODEL = "Qwen/Qwen3-0.6B"

    def __init__(self) -> None:
        """Initialize the model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.MODEL, torch_dtype="auto", device_map=self.device
        )
        self.model.eval()

    def answer(self, question: str, context: str, max_new: int = 256) -> str:
        """Generate the answer from context."""
        msgs = [
            {"role": "system", "content": "Answer only from the context."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new,
                                  do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]  # drop the echoed prompt
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return str(text).strip()
