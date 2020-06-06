from kospeech.model.encoder import Listener
from kospeech.model.decoder import Speller
from kospeech.model.seq2seq import ListenAttendSpell

listener = Listener(80, 256, 'cpu')
speller = Speller(2038, 151, 512, 1, 2)
model = ListenAttendSpell(listener, speller)

print(model)