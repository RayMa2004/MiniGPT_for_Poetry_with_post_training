import json
from collections import Counter
from pathlib import Path

class CharTokenizer:
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    # PAD : padding token

    def __init__(self, stoi = None):
        # stoi : str to int
        self.stoi = stoi or {}
        # stoi[str] = int for encode
        # dist.items() --> a sequence like (key, value)
        # itos[int] --> str for decode
        self.itos = {idx : ch for ch, idx in self.stoi.items()}

    @classmethod  # this is a class method
    def build(cls, texts, min_freq = 1): 
        # cls : class
        # min_freq : the minimum frequency of a char
        counter = Counter()

        for text in texts:
            counter.update(text)

        # assign token id for special token
        stoi = {
            token : i
            for i, token in enumerate(cls.SPECIAL_TOKENS)
        }

        # making chars sorted in order to make vocabulary consistent
        chars = sorted(
            ch for ch, freq in counter.items()
            if freq >= min_freq and ch not in stoi
        )

        for ch in chars:
            stoi[ch] = len(stoi) # a fancy idea

        return cls(stoi) # == return CharTokenizer(stoi)

    @property 
    # property : 
    #     "tokenizer.vocab_size" rather than "tokenizer.vocab_size()"
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_id(self):
        return self.stoi["<PAD>"]

    @property
    def bos_id(self):
        return self.stoi["<BOS>"]

    @property
    def eos_id(self):
        return self.stoi["<EOS>"]

    @property
    def unk_id(self):
        return self.stoi["<UNK>"]
    
    def encode(self, text, add_special_tokens = True):
        ids = [
            self.stoi.get(ch, self.stoi["<UNK>"])
            # get(key, default)
            for ch in text
        ]

        if add_special_tokens:
            # <BOS> + tokens + <EOS>
            ids = [self.bos_id] + ids +[self.eos_id]

        return ids

    def decode(self, idxs, skip_special_tokens = True):
        chars = []

        for idx in idxs:
            ch = self.itos.get(int(idx),"<UNK>")

            if skip_special_tokens and ch in self.SPECIAL_TOKENS:
                continue

            chars.append(ch)

        return "".join(chars)

    def save(self, path):
        # with : close file automatically after operation
        # w : writing mode
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        # r : reading mode
        with open(path, "r", encoding='utf-8') as f:
            stoi = json.load(f)

        return cls(stoi)
