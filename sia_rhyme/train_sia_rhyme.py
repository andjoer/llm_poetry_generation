from sia_rhyme.siamese_rhyme import siamese_rhyme


model = siamese_rhyme(load=False)

model.save_vocabs()
model.train()
print('predict')
print(model.predict('haus','maus'))
print(model.predict('haus','ganz'))

print(model.predict('gehen','sacken'))

print(model.predict('katze','atzek'))


print(model.predict('gehen','standen'))