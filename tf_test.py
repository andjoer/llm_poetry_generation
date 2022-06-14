#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



#text = ''' Kommt, laßt uns heut' ihn uns [MASK] [MASK] – und finden wir ihn nicht,'''

'''from transformers import pipeline
top_k = 2
unmasker = pipeline('fill-mask', model = 'Anjoe/german-poetry-distilbert', top_k = top_k,framework='pt')
#unmasker = pipeline('fill-mask', model = 'xlm-roberta-large-finetuned-conll03-german', top_k = top_k,framework='pt')




predictions = unmasker(text)

print(predictions)
for prediction in predictions:

      print(prediction[0]['token_str'])
      print(prediction[1]['token_str'])'''


from rythm import verse_cl

from rythm import fix_rythm



verse = verse_cl('''Wichtigkeit nach für weit vorzüglicher und ihre Endabsicht für viel erhabener
halten als alles, was der Verstand im Felde der Erscheinungen
lernen kann, wobei wir sogar auf die Gefahr zu irren eher alles wagen,
als daß wir so angelegene Untersuchungen aus irgend einem Grunde der
Bedenklichkeit, oder aus Geringschätzung und Gleichgültigkeit aufgeben sollten.''') 

fix_rythm(verse,[0,1],9)





