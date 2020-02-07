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
        self.cumulative_p, self.candidates = init_step_output.topk(self.k)
        speller_input = self.candidates
        self.candidates = self.candidates.view(self.batch_size, self.k, 1)

        for di in range(self.max_len-1):
            predicted_softmax = self.forward_step(speller_input, listener_outputs)
            # (batch_size, k, classfication_num)
            # 빔 개수만큼의 2,040개에 대한 분포를 구한다.
            step_output = predicted_softmax.squeeze(1)
            # (batch_size, k, k)
            # 2,040개의 분포 중 상위 K개를 뽑는다
            candidate_p, candidate_k = step_output.topk(self.k)
            # (batch_size, k * k)
            # K개의 입력에서, K개의 출력이 각각 나왔으므로, K * K 형상으로 바꿔준다.
            # Ex) 3개의 빔에서 3개의 아웃풋이 나왔다면 결국은, 9개의 확률이 나온 것.
            # Log-probability라 덧셈으로 !!
            candidate_p = (self.cumulative_p + candidate_p).view(self.batch_size, self.k * self.k)
            candidate_k = candidate_k.view(self.batch_size, self.k * self.k)
            # (batch_size, k)
            # K * K개 중 K개를 고른다.
            # Ex) 앞에서 나온 9개의 확률 중 큰 3개 (빔사이즈) 를 고른다.
            select_p, select_idx = candidate_p.topk(self.k)
            # (batch_size, k)
            select_k = torch.LongTensor(self.batch_size, self.k)
            # (batch_size, k, seq_len)
            candidates = torch.LongTensor(self.candidates.size(0), self.candidates.size(1), self.candidates.size(2))
            parent_node = (select_idx % self.k).view(self.batch_size, self.k)
            for batch_num, batch in enumerate(select_idx):
                for idx, value in enumerate(batch):
                    select_k[batch_num, idx] = candidate_k[batch_num, value]
                    candidates[batch_num, idx] = self.candidates[batch_num, parent_node[batch_num, idx]]
            self.candidates = torch.cat([candidates, select_k.view(self.batch_size, self.k, 1)], dim=2)
            speller_input = select_k
            """ eos 처리 & Length Penalty 추가해야함 """

    def forward_step(self, speller_input, listener_outputs):
        output_size = speller_input.size(1)
        embedded = self.embedding(speller_input)
        embedded = self.input_dropout(embedded)

        speller_output, hidden = self.rnn(embedded, self.speller_hidden)  # speller output

        if self.use_attention:
            output = self.attention(decoder_output=speller_output, encoder_output=listener_outputs)
        else: output = speller_output
        # torch.view()에서 -1이면 나머지 알아서 맞춰줌
        predicted_softmax = self.decode_func(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(self.batch_size,output_size,-1)
        return predicted_softmax