from torchtext import data

def pool(d, random_shuffler, batch_size, sort_key):
    for p in data.batch(d, batch_size * 100):
        p_batch = data.batch(
            sorted(p, key=sort_key),
            batch_size
        )
        for b in random_shuffler(list(p_batch)):
            yield b

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            self.batches = pool(
                self.data(), 
                self.random_shuffler, 
                self.batch_size,
                self.sort_key
            )
        else:
            self.batches = []
            batches = data.batch(
                self.data(), 
                self.batch_size
            )
            for b in batches: 
                self.batches.append(sorted(b, key=self.sort_key))