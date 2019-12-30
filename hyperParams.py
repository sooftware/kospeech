class HyperParams():
    def __init__(self):
        self.bidirectional = False
        self.attention = False
        self.hidden_size = 256
        self.dropout = 0.5
        self.encoder_layer_size = 6
        self.decoder_layer_size = 3
        self.batch_size = 1
        self.workers = 4
        self.max_epochs = 30
        self.lr = 0.0001
        self.teacher_forcing = 0.8
        self.seed = 1
        self.max_len = 80
        self.no_cuda = True
        self.save_name = 'model'
        self.mode = 'train'

    def input_params(self):
        use_bidirectional = input("use bidirectional : ")
        if use_bidirectional.lower() == 'true':
            self.bidirectional = True
        use_attention = input("use_attention : ")
        if use_attention.lower() == 'true':
            self.attention = True
        self.hidden_size = int(input("hidden_size : "))
        self.dropout = float(input("dropout : "))
        self.encoder_layer_size = int(input("encoder_layer_size : "))
        self.decoder_layer_size = int(input("decoder_layer_size : "))
        self.batch_size = int(input("batch_size : "))
        self.workers = int(input("workers : "))
        self.max_epochs = int(input("max_epochs : "))
        self.lr = float(input("learning rate : "))
        self.teacher_forcing = float(input("teacher_forcing : "))
        self.seed = int(input("seed : "))