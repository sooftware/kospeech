from kospeech.models.seq2seq.encoder import Seq2seqEncoder
from kospeech.models.seq2seq.decoder import Seq2seqDecoder
from kospeech.models.seq2seq.seq2seq import Seq2seq

encoder = Seq2seqEncoder(161, 512, 'cpu')
decoder = Seq2seqDecoder(2038, 151, 1024, 1, 2)
model = Seq2seq(encoder, decoder)

print(model)

"""
listener = Listener(80, 256, 'cpu', extractor='ds2')
speller = Speller(2038, 151, 512, 1, 2, attn_mechanism='loc')
model = ListenAttendSpell(listener, speller)

print(model)
"""