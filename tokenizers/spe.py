import unicodedata
from collections import defaultdict, Counter
import json

class SP_BPE:
    class _Node:
        def __init__(self, value, prev=None, next=None):
            self.value = value
            self.prev = prev
            self.next = next

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.merge_ranks = {}
        self._inverse_vocab = {}
        
        # Define special tokens as class attributes for clarity
        self.WHITESPACE_CHAR = ' '
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.BOS_ID = 2
        self.EOS_ID = 3

    def _normalize(self, text):
        """Normalizes text and replaces spaces with a special character."""
        normalized_text = unicodedata.normalize('NFKC', text).lower()
        return normalized_text.replace(' ', self.WHITESPACE_CHAR)

    def train(self, text, size):
        """Trains the BPE tokenizer on a given text corpus."""
        if size < 260:
            raise ValueError("Vocab size must be at least 260 to cover special tokens and base bytes.")

        self.vocab = {
            self.PAD_ID: b'<pad>',
            self.UNK_ID: b'<unk>',
            self.BOS_ID: b'<s>',
            self.EOS_ID: b'</s>',
            **{i + 4: bytes([i]) for i in range(256)}
        }

        text_bytes = self._normalize(text).encode('utf-8')

        # Create a double linked list for efficient merging
        d_head = self._Node(None)
        prev_node = d_head
        for byte_val in text_bytes:
            new_node = self._Node(byte_val + 4, prev=prev_node)
            prev_node.next = new_node
            prev_node = new_node
        head = d_head.next
        if head:
            head.prev = None

        # Calculate initial pair frequencies
        freqs = Counter()
        p_nodes = defaultdict(set)
        node = head
        while node and node.next:
            pair = (node.value, node.next.value)
            freqs[pair] += 1
            p_nodes[pair].add(node)
            node = node.next

        num_merges = size - len(self.vocab)
        next_id = len(self.vocab)

        for i in range(num_merges):
            if not freqs:
                break
            
            best_pair = max(freqs, key=freqs.get)

            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            process_nodes = list(p_nodes.pop(best_pair))
            del freqs[best_pair]
            processed_nodes = set()

            for x in process_nodes:
                if x in processed_nodes or (x.next and x.next in processed_nodes):
                    continue
                if not x.next or (x.value, x.next.value) != best_pair:
                    continue
                
                prev_node = x.prev
                next_node = x.next
                next_next_node = next_node.next

                def update_stats(pair, modifying_node, delta):
                    if not pair: return
                    freqs[pair] += delta
                    if delta > 0: p_nodes[pair].add(modifying_node)
                    else: p_nodes[pair].discard(modifying_node)
                    if freqs[pair] <= 0:
                        del freqs[pair]
                        if pair in p_nodes: del p_nodes[pair]

                update_stats((prev_node.value, x.value) if prev_node else None, prev_node, -1)
                update_stats((next_node.value, next_next_node.value) if next_next_node else None, next_node, -1)

                x.value = next_id
                x.next = next_next_node
                if next_next_node: next_next_node.prev = x
                
                processed_nodes.add(x)
                processed_nodes.add(next_node)
                
                update_stats((prev_node.value, x.value) if prev_node else None, prev_node, 1)
                update_stats((x.value, next_next_node.value) if next_next_node else None, x, 1)

            next_id += 1
            if (i + 1) % 500 == 0:
                print(f"Merge {i+1}/{num_merges} complete.")
        
        print("Training complete.")
        self._build_datastructures()

    def _build_datastructures(self):
        """Builds helper data structures for fast encoding."""
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        """Encodes a string into a list of token IDs."""
        if not self.merges:
            raise RuntimeError("Tokenizer has not been trained yet.")

        normalized_text = self._normalize(text)
        text_bytes = normalized_text.encode('utf-8')
        
        tokens = [byte_val + 4 for byte_val in text_bytes]

        while len(tokens) > 1:
            pairs = {(tokens[i], tokens[i+1]): i for i in range(len(tokens) - 1)}
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

            if best_pair not in self.merges:
                break

            idx = pairs[best_pair]
            new_id = self.merges[best_pair]
            tokens = tokens[:idx] + [new_id] + tokens[idx+2:]
        
        if add_special_tokens:
            return [self.BOS_ID] + tokens + [self.EOS_ID]
        return tokens

    def decode(self, ids):
        """Decodes a list of token IDs back into a string."""
        bytes_chunks = [self.vocab.get(id, self.vocab[self.UNK_ID]) for id in ids]
        text_bytes = b"".join(bytes_chunks)
        text = text_bytes.decode('utf-8', 'replace').replace(self.WHITESPACE_CHAR, ' ')
        return text

    def save(self, prefix):
        """Saves the tokenizer's state (vocab and merges) to files."""
        vocab_file = f"{prefix}_vocab.json"
        merges_file = f"{prefix}_merges.json"

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json_vocab = {k: list(v) for k, v in self.vocab.items()}
            json.dump(json_vocab, f, ensure_ascii=False, indent=2)

        with open(merges_file, 'w', encoding='utf-8') as f:
            json_merges = {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
            json.dump(json_merges, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {vocab_file} and {merges_file}")

    @classmethod
    def load(cls, prefix):
        """Loads a tokenizer from saved vocab and merges files."""
        tokenizer = cls()
        vocab_file = f"tokenizers/{prefix}_vocab.json"
        merges_file = f"tokenizers/{prefix}_merges.json"
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            json_vocab = json.load(f)
            tokenizer.vocab = {int(k): bytes(v) for k, v in json_vocab.items()}

        with open(merges_file, 'r', encoding='utf-8') as f:
            json_merges = json.load(f)
            tokenizer.merges = {tuple(map(int, k.split(','))): v for k, v in json_merges.items()}

        tokenizer._build_datastructures()
        print(f"Tokenizer loaded successfully from '{prefix}' files.")
        return tokenizer