import torch

class Beam:
    def __init__(self, k, speller_input, speller_hidden,
                 batch_size, max_len, decode_func, rnn,
                 embedding, input_dropout, use_attention, attention,
                 hidden_size, out):
        self.k = k
        self.speller_input = speller_input
        self.speller_hidden = speller_hidden
        self.batch_size = batch_size
        self.max_len = max_len
        self.decode_func = decode_func
        self.rnn = rnn
        self.embedding = embedding
        self.input_dropout = input_dropout
        self.use_attention = use_attention
        self.attention = attention
        self.hidden_size = hidden_size
        self.out = out
        self.cumulative_p = None
        self.candidates = None

    def search(self, init_speller_input, listener_outputs):
        """
        listener_outputs : (batch_size, seq_len, hidden_size)
        """
        init_predicted_softmax = self.forward_step(init_speller_input, listener_outputs)
        # (batch_size, classification_num)
        init_step_output = init_predicted_softmax.squeeze(1)
        # top value & top k
        # (batch_size, k)
        init_topv, init_topk = init_step_output.topk(self.k)
        self.cumulative_p = init_topv
        self.candidates = speller_input = init_topk
        self.candidates = self.candidates.view(self.batch_size, self.k, 1)

        for di in range(self.max_len):
            predicted_softmax = self.forward_step(speller_input, listener_outputs)
            # (batch_size, classfication_num)
            step_output = predicted_softmax.squeeze(1)
            candidate_p, candidate_k = step_output.topk(self.k)
            print(candidate_k.size())
            print(candidate_k)
            element_p = (self.cumulative_p * candidate_p).view(self.batch_size, self.k * self.k)
            select_p, test = element_p.topk(self.k)
            print(test.size())
            print(test)
            select_k = candidate_k.view(self.batch_size, self.k * self.k)
            self.cumulative_p *= select_p # 확률쪽 여기 수정
            self.candidates = torch.cat([self.candidates, select_k.view(self.batch_size, self.k, 1)], dim=2)
            # 이제 추가까지 됐는데 상단 노드랑 어떻게 매칭을 시켜줄 것인가,,,

            speller_input = select_k
            if di > 10:
                exit()

    def forward_step(self, speller_input, listener_outputs):
        output_size = speller_input.size(1)  # speller_input.size(1) : seq_len
        embedded = self.embedding(speller_input)
        embedded = self.input_dropout(embedded)

        speller_output, hidden = self.rnn(embedded, self.speller_hidden)  # speller output

        if self.use_attention:
            output = self.attention(decoder_output=speller_output, encoder_output=listener_outputs)
        else: output = speller_output
        # torch.view()에서 -1이면 나머지 알아서 맞춰줌
        predicted_softmax = self.decode_func(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(self.batch_size,output_size,-1)
        return predicted_softmax

