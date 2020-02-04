

class Beam:
    def __init__(self, batch_size, eos_id):
        self.eos_id = eos_id
        self.symbols = []
        self.probs = []

    def step_forward(self):
        pass