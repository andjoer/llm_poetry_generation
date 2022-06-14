
#ortho = torch.load('vocab/ortho.pt')
#ipa = torch.load('vocab/ipa.pt')

from ortho_to_ipa import ortho_to_ipa

#gti = ortho_to_ipa(fname_model='checkpoints/seq2seq_212630.pt')

gti = ortho_to_ipa(load = True,fpath='data')

#gti.save_vocabs()
#gti.train()

print(gti.translate('erstorbnem'))

#dictionary = gti.get_vocab()

#print(dictionary)


#pred = gti.get_vectors('hallo')
#print(pred)

#pred = gti.translate('ibims')
#print(pred)