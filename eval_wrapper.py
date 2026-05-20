import torch
import torch.nn.functional as F
from lm_eval.api.model import LM


"""
Wrapper class to interface the custom transformer model with the
Language Model Evaluation Harness (lm-eval) framework
Implements the required LM API methods to enable standardised
evaluation across benchmarks such as BLiMP, HellaSwag, and WinoGrande
"""
class TransformerEvalWrapper(LM):
    def __init__(self, model, tokenizer, device, batch_size=1):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = batch_size
        self.model.eval()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size  # 50257 for GPT-2

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)

    """
    Computes the log likelihood of a continuation given a context for
    each request
    Tokenises the context and continuation, constructs
    the full input sequence, and extracts the logits corresponding to
    the continuation tokens
    Returns the total log probability of the
    continuation and a greedy flag indicating whether the model would
    have predicted the continuation greedily
    """
    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            if len(ctx_ids) == 0:
                ctx_ids = [self.eot_token_id]
            if len(ctx_ids) + len(cont_ids) > self.max_length:
                ctx_ids = ctx_ids[-(self.max_length - len(cont_ids)):]

            input_ids = torch.tensor(
                [ctx_ids + cont_ids], dtype=torch.long
            ).to(self._device)

            assert input_ids.max().item() < self.vocab_size, \
                f"Token ID {input_ids.max().item()} exceeds vocab_size {self.vocab_size}"
            assert input_ids.size(1) <= self.max_length, \
                f"Sequence length {input_ids.size(1)} exceeds max_seq_length {self.max_length}"

            with torch.no_grad():
                logits = self._model_call(input_ids)  

            cont_len = len(cont_ids)
            ctx_len = len(ctx_ids)

            cont_logits = logits[0, ctx_len - 1 : ctx_len + cont_len - 1, :]  
            cont_targets = input_ids[0, ctx_len : ctx_len + cont_len]         

            assert cont_logits.shape[0] == cont_len, \
                f"Logit slice mismatch: {cont_logits.shape[0]} != {cont_len}"
            assert (cont_targets < self.vocab_size).all(), \
                f"Target token out of vocab range: {cont_targets.max().item()}"

            log_probs = F.log_softmax(cont_logits, dim=-1)
            token_log_probs = log_probs[torch.arange(cont_len), cont_targets]
            total_log_prob = token_log_probs.sum().item()
            is_greedy = (cont_logits.argmax(dim=-1) == cont_targets).all().item()
            results.append((total_log_prob, is_greedy))
        return results
        
        
    """
    Computes the rolling log likelihood over an entire text sequence
    for each request
    Tokenises the full text, runs a single forward
    pass, and computes the log probability of each token given all
    preceding tokens
    Returns the total log probability summed across
    all tokens in the sequence.
    """

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)

            if len(token_ids) > self.max_length:
                token_ids = token_ids[-self.max_length:]

            input_ids = torch.tensor(
                [token_ids], dtype=torch.long
            ).to(self._device)

            assert input_ids.max().item() < self.vocab_size, \
                f"Token ID {input_ids.max().item()} exceeds vocab_size {self.vocab_size}"

            with torch.no_grad():
                logits = self._model_call(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)
            targets = input_ids[0, 1:]
            scores = log_probs[0, :-1, :]
            token_log_probs = scores[torch.arange(len(targets)), targets]
            results.append(token_log_probs.sum().item())
        return results

    def generate_until(self, requests):
        raise NotImplementedError