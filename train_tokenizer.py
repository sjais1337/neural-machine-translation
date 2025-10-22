from tokenizers.spe import SP_BPE


def train_my_tokenizer():
    combined_corpus_path = "data/merged.txt"

    with open(combined_corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    vocab_size = 8000

    tokenizer = SP_BPE()
    tokenizer.train(corpus_text, vocab_size)

    save_prefix = "nmt_8000"
    tokenizer.save(save_prefix)


if __name__ == "__main__":
    train_my_tokenizer()
