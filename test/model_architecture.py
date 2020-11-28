from kospeech.models.las.encoder import Listener
from kospeech.models.las.decoder import Speller
from kospeech.models.las.model import ListenAttendSpell
from kospeech.models.transformer.model import SpeechTransformer


encoder = Listener(80, 512, 'cpu')
decoder = Speller(2038, 151, 1024, 1, 2)
model = ListenAttendSpell(encoder, decoder)

print(model)


model = SpeechTransformer(num_classes=2038, num_encoder_layers=3, num_decoder_layers=3)
print(model)
