from torchtext import data

def pool(d, random_shuffler, batch_size, sort_key, batch_size_fn):
    for p in data.batch(d, batch_size * 100):
        p_batch = data.batch(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn
        )
        for b in random_shuffler(list(p_batch)):
            yield b

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, _):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = pool(
                self.data(), 
                self.random_shuffler, 
                self.batch_size,
                self.sort_key,
                self.batch_size_fn
            )
        else:
            self.batches = []
            batches = data.batch(
                self.data(), 
                self.batch_size,
                self.batch_size_fn
            )
            for b in batches: 
                self.batches.append(sorted(b, key=self.sort_key))