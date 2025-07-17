import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class LlamaTCompletionRequest(BaseModel):
    """Request model for LlamaT completion."""
    messages: List[Message]
    temperature: Optional[float] = 0.0
    max_new_tokens: Optional[int] = 128


class LlamaTPrompter:
    def __init__(self, model_dir="local_models/llamat-2-chat"):
        # Check for multiple GPUs
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_gpus = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_gpus = 1
        else:
            self.device = torch.device("cpu")
            self.num_gpus = 1

        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        
        # Move model to device and handle multiple GPUs
        if self.device.type == "mps":
            self.model = self.model.half()
        elif self.num_gpus > 1:
            # Use DataParallel for multiple GPUs
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        # Get special token IDs
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _extract_response(self, full_text):
        """Extract response from generated text"""
        # truncate at first <|im_end|> so echoes never leak
        reply = full_text.split("<|im_end|>")[2].split("assistant", 1)[-1].strip()
        return reply

    def generate(self, messages: List[Message], temperature: float = 0.0, max_new_tokens: int = 128):
        """Generate response from messages"""
        # Build conversation prompt
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|im_start|>system\n{msg.content}\n<|im_end|>\n"
            elif msg.role == "user":
                prompt += f"<|im_start|>user\n{msg.content}\n<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{msg.content}\n<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        # Get the actual model (handle DataParallel case)
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

        # Generate text
        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = self._extract_response(generated_text)
        
        # Calculate token usage
        total_tokens = outputs[0].shape[0]
        new_tokens = total_tokens - input_length
        
        metadata = {
            "promptTokenCount": input_length,
            "candidatesTokenCount": new_tokens,
            "totalTokenCount": total_tokens
        }
        
        return response, metadata


# Global prompter instance
_prompter = None


def _get_prompter():
    """Get or create global prompter instance"""
    global _prompter
    if _prompter is None:
        _prompter = LlamaTPrompter()
    return _prompter


def llamat_text_completion(in_request: LlamaTCompletionRequest) -> tuple[str, Dict[str, Any]]:
    """
    Make a LlamaT completion call.
    Returns response text and metadata similar to other LLM functions.
    """
    try:
        prompter = _get_prompter()
        response, metadata = prompter.generate(
            messages=in_request.messages,
            temperature=in_request.temperature,
            max_new_tokens=in_request.max_new_tokens
        )
        return response, metadata
    
    except Exception as e:
        raise RuntimeError(f"LlamaT generation failed: {str(e)}")


def simple_llamat_call(prompt: str, system_prompt: str = "", max_new_tokens: int = 128) -> str:
    """
    Simple function to call LlamaT with a basic prompt.
    Returns just the response text.
    """
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=prompt))
    
    request = LlamaTCompletionRequest(messages=messages, max_new_tokens=max_new_tokens)
    response, _ = llamat_text_completion(request)
    return response
