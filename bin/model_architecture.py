from kospeech.models.acoustic.seq2seq.encoder import SpeechEncoderRNN
from kospeech.models.acoustic.seq2seq.decoder import SpeechDecoderRNN
from kospeech.models.acoustic.seq2seq.seq2seq import SpeechSeq2seq

encoder = SpeechEncoderRNN(80, 512, 'cpu')
decoder = SpeechDecoderRNN(2038, 151, 1024, 1, 2)
model = SpeechSeq2seq(encoder, decoder)

print(model)
