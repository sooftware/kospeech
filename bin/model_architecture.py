from kospeech.models.las.encoder import SpeechEncoderRNN
from kospeech.models.las.decoder import SpeechDecoderRNN
from kospeech.models.las.las import SpeechSeq2seq
from kospeech.models.transformer.transformer import SpeechTransformer


encoder = SpeechEncoderRNN(80, 512, 'cpu')
decoder = SpeechDecoderRNN(2038, 151, 1024, 1, 2)
model = SpeechSeq2seq(encoder, decoder)

print(model)


model = SpeechTransformer(num_classes=2038, num_encoder_layers=3, num_decoder_layers=3)
print(model)
