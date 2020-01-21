from loader.baseLoader import BaseDataLoader

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_num):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_num = worker_num
        self.loader = list()

        for idx in range(self.worker_num):
            self.loader.append(BaseDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        # BaseDataLoader run()을 실행!!
        for idx in range(self.worker_num):
            self.loader[idx].start()

    def join(self):
        for idx in range(self.worker_num):
            self.loader[idx].join()