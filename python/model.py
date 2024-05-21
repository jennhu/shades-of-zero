from surprisal import AutoHuggingFaceModel


class LM(object):
    """
    Wrapper class around AutoHuggingFaceModel from the `surprisal` library.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        # NOTE: "model_class" needs to be "gpt" for causal LMs,
        # but the tokenizer and model will still be loaded according to `model_name`
        # See https://github.com/aalok-sathe/surprisal/issues/19
        self.m = AutoHuggingFaceModel.from_pretrained(
            model_name, 
            model_class="gpt"
        )
        try:
            self.m.to("cuda")
            self.device = "cuda"
        except:
            self.device = "cpu"

    def _agg_surprisal(self, token_surprisals, start, end, level="char"):
        sum_surp = token_surprisals[start:end, level]
        return sum_surp

    def _get_token_surprisals(self, text):
        [token_surprisals] = self.m.surprise(text)
        return token_surprisals

    def get_surprisal_of_continuation(self, prefix, continuation, sep=" "):
        text = prefix + sep + continuation
        token_surprisals = self._get_token_surprisals(text)
        
        n_prefix_chars = len(prefix)
        n_continuation_chars = len(continuation)
        return self._agg_surprisal(
            token_surprisals, 
            n_prefix_chars, 
            n_prefix_chars + n_continuation_chars + 1,
            level="char"
        )
