import random


def linelen(line):
    "num tokens + 1 for eos"
    return len(line.split()) + 1


def maxlen(item):
    return max(linelen(l) for l in item if isinstance(l, str))


def cycle_shuffle(iterator):
    items = list(iterator)
    while True:
        random.shuffle(items)
        yield from items


def form_batches(data, batch_size):
    seq = iter(data)
    done = False
    while not done:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(seq))
            except StopIteration:
                done = True
        if batch:
            yield batch


def form_adaptive_batches(data, batch_cost, cost_func=maxlen, batch_size_max=0):
    seq = iter(data)
    prev = []
    max_len = 0
    done = False
    while not done:
        batch = prev
        try:
            while True:
                item = next(seq)
                max_len = max(max_len, cost_func(item))
                if (len(batch) + 1) * max_len > batch_cost or (batch_size_max and len(batch) >= batch_size_max):
                    prev, max_len = [item], cost_func(item)
                    break
                batch.append(item)
        except StopIteration:
            done = True
        if batch:
            yield batch


def form_adaptive_batches_windowed(data, cost_func=maxlen, sort_key_func=None,
                                   max_size=5000, split_len=10000, batch_size_max=0):
    rng = random.Random(42)
    buf = []
    last_chunk = []
    reverse = False
    for p in data:
        if len(buf) >= split_len:
            # Last chunk may contain fewer sentences than others - let's return in to the miller
            buf += last_chunk

            buf = sorted(buf, key=sort_key_func or cost_func, reverse=reverse)
            chunks = list(form_adaptive_batches(buf, max_size, cost_func=cost_func, batch_size_max=batch_size_max))

            last_chunk = chunks.pop()
            buf = []

            reverse = not reverse

            rng.shuffle(chunks)
            for chunk in chunks:
                yield chunk
        buf.append(p)

    buf += last_chunk
    buf = sorted(buf, key=sort_key_func or cost_func, reverse=reverse)
    chunks = list(form_adaptive_batches(buf, max_size, cost_func=cost_func, batch_size_max=batch_size_max))
    rng.shuffle(chunks)
    for chunk in chunks:
        yield chunk


def filter_by_len(data, max_srclen=None, max_dstlen=None, batch_len=None):
    def item_ok(item):
        return ((max_srclen is None or linelen(item[0]) <= max_srclen) and
                (max_dstlen is None or linelen(item[0]) <= max_dstlen) and
                (batch_len is None or maxlen(item) <= batch_len))
    return filter(item_ok, data)
