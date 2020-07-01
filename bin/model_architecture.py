from kospeech.model.encoder import Listener
from kospeech.model.decoder import Speller
from kospeech.model.las import ListenAttendSpell

listener = Listener(161, 512, 'cpu')
speller = Speller(2038, 151, 1024, 1, 2)
model = ListenAttendSpell(listener, speller)

print(model)

listener = Listener(80, 256, 'cpu', extractor='ds2')
speller = Speller(2038, 151, 512, 1, 2, attn_mechanism='loc')
model = ListenAttendSpell(listener, speller)

print(model)
