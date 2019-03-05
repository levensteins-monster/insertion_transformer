import random
import bintrees
import threading
from itertools import count
from collections import Counter
import tensorflow as tf
from joblib import Parallel, delayed
from lib.ops import get_available_gpus
from lib.trainer import SampleBasedTrainer


class MultiGPUTrainer:
    def __init__(self, name, make_model,
                 devices=get_available_gpus(),
                 master_device=None,
                 TrainerClass=SampleBasedTrainer,
                 sess=None, *args, verbose=False, **kwargs):
        """ A wrapper-class that performs batch-parallel training with some trainer. """

        self.name = name
        self.sess = sess = sess or tf.get_default_session() or tf.InteractiveSession()
        self.master_device = master_device = master_device or next(iter(devices))
        assert master_device in devices
        self.verbose = verbose

        class Worker(TrainerClass):
            def get_optimizer(self, *args, **kwargs):
                """ Worker does not update weights by itself. use sgd to avoid wasting memory """
                return tf.train.GradientDescentOptimizer(learning_rate=0)

        with tf.variable_scope(name):
            self.workers_by_device = {}
            for i, device in enumerate(devices):
                with tf.device(device), tf.variable_scope('worker_%i' % i):
                    model = make_model()
                    if device == master_device:
                        worker = TrainerClass(model, *args, **kwargs)
                    else:
                        worker = Worker(model, *args, **kwargs)
                    self.workers_by_device[device] = worker

                if verbose:
                    print("Created model {} weights and worker on device {}"
                          "".format(model.name, device))

        self.master_model = self.workers_by_device[master_device].model
        self.master_worker = self.workers_by_device[self.master_device]
        assert isinstance(self.master_worker, TrainerClass)

        # step 1: send main model's weights to all worker replicas
        self.scatter_weights = []

        for device, worker in self.workers_by_device.items():
            if worker == self.master_worker:
                continue
            self.scatter_weights.extend(map(tf.assign,
                                            worker.optimized_variables,
                                            self.master_worker.optimized_variables))

        # step 2: compute grads and counters at all workers
        self.gather_grads, self.gather_counters = [], []
        for device, worker in self.workers_by_device.items():
            if worker == self.master_worker:
                continue
            self.gather_grads.extend(
                map(tf.assign_add, self.master_worker.accumulated_grads, worker.accumulated_grads)
            )
            self.gather_grads.append(
                tf.assign_add(self.master_worker.accumulated_num_batches, worker.accumulated_num_batches)
            )
            master_counters_flat = [self.master_worker.accumulated_counters[name]
                                    for name in sorted(self.master_worker.accumulated_counters.keys())]
            worker_counters_flat = [worker.accumulated_counters[name]
                                    for name in sorted(self.master_worker.accumulated_counters.keys())]
            self.gather_counters.extend(
                map(tf.assign_add, master_counters_flat, worker_counters_flat)
            )

        # step 3: perform gradient step and reset all accumulated values
        self.reset_slave_grads = [
            worker.reset_gradients for worker in self.workers_by_device.values()
            if worker != self.master_worker
        ]
        self.reset_slave_counters = [
            worker.reset_counters for worker in self.workers_by_device.values()
            if worker != self.master_worker
        ]

    def train_on_batches(self, batches, optimizer_step=True, reset_counters=None, **kwargs):
        sess = self.sess
        lock = threading.Lock()
        available_devices = set(self.workers_by_device.keys())

        def _thread(batch):
            with lock:
                assert len(available_devices) != 0, "all devices busy. this should't ever happen"
                device = available_devices.pop()
                if self.verbose:
                    print("thread {} acquired device {}".format(threading.get_ident(), device), flush=True)
            result = self.workers_by_device[device].train_on_batch(batch, optimizer_step=False,
                                                                   reset_counters=False, **kwargs)
            with lock:
                if self.verbose:
                    print("thread {} released device {}".format(threading.get_ident(), device), flush=True)
                available_devices.add(device)
            return result

        tasks = [delayed(_thread)(batch) for batch in batches]
        _ = Parallel(backend='threading', n_jobs=len(self.workers_by_device))(tasks)

        sess.run(self.gather_counters)
        metrics = sess.run(self.master_worker.compute_metrics)

        if optimizer_step:
            sess.run(self.gather_grads)
            sess.run(self.master_worker.apply_gradients)
            sess.run([self.master_worker.reset_gradients, self.reset_slave_grads, self.scatter_weights])
        if reset_counters is None:
            reset_counters = optimizer_step
        if reset_counters:
            sess.run(self.master_worker.reset_counters)

        return metrics


class ParallelBatchIterator:
    def __init__(self, iterator, cost_func, n_buffers, random_state=42, **kwargs):
        """
        groups iterator items (batches) by cost and returns tuples of batches with
        approximtely the same cost. Uses cost buffer
        :param iterator: iterator over batches
        :param cost_func: lambda batch -> float cost,
            typically it's approximate time it takes to process batch
        :param n_buffers: number of batches to return in parallel.
        """
        self.available_costs, self.cost_counts = set(), Counter()
        self.cost_func = cost_func

        def iterate_with_costs(iterator):
            for batch in iterator:
                cost = cost_func(batch)
                self.available_costs.add(cost)
                self.cost_counts[cost] += 1
                yield batch, cost

        self.cost_buffers = [
            self.CostBuffer(iterate_with_costs(iterator)) for _ in range(n_buffers)
        ]
        self.rng = random.Random(42)

    def __iter__(self):
        return self

    def __next__(self):
        # sample random existing cost
        sample_cost = self.rng.sample(self.available_costs, 1)[0] if len(self.available_costs) else 0
        batches, costs = zip(*(cost_buffer.pop(sample_cost)
                               for cost_buffer in self.cost_buffers))
        for cost in costs:
            self.cost_counts[cost] -= 1
            if self.cost_counts[cost] <= 0:
                self.cost_counts.pop(cost)
                self.available_costs.remove(cost)
        return batches

    class CostBuffer:
        def __init__(self, iterator_with_costs, buf_size=1000):
            """
            A tool for quickly finding batches with approximately specified cost.
            Used for multi-gpu batch balancing
            Credits: Yandex MT team
            """
            self.iterator_with_costs = iterator_with_costs
            self.current_size, self.max_size = 0, buf_size
            self.tree = bintrees.FastRBTree()

        def pop(self, target_cost):
            """ Samples batch near target_cost. """
            # warmup: read from iterator
            for batch, cost in self.iterator_with_costs:
                if cost in self.tree:
                    self.tree[cost].append(batch)
                else:
                    self.tree[cost] = [batch]
                self.current_size += 1
                if self.current_size >= self.max_size:
                    break

            if not len(self.tree):
                raise StopIteration("No elements left in buffer/iterator")

            # generate cost to choose and choose relevant batch
            cost = self._find_nearest_cost(target_cost)
            batch = self.tree[cost][0]

            # remove selected items from structures
            del self.tree[cost][0]
            if len(self.tree[cost]) == 0:
                del self.tree[cost]
            self.current_size -= 1

            return batch, cost

        def _find_nearest_cost(self, cost):
            cost = max(cost, self.tree.min_key())
            cost = min(cost, self.tree.max_key())
            floor_cost = self.tree.floor_key(cost)
            ceil_cost = self.tree.ceiling_key(cost)
            return floor_cost if abs(ceil_cost - cost) < abs(floor_cost - cost) else ceil_cost
