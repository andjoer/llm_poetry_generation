from sia_rhyme.siamese_rhyme import siamese_rhyme


model = siamese_rhyme()


print('predict')


print(model.get_distance('katze','klaget'))

vector1 = model.get_word_vec('waldlein')
vector2 = model.get_word_vec('einmal')

print(model.get_distance_vec('mähen',vector1))
print(model.vector_distance(vector1,vector2))