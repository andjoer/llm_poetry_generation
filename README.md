# Poetry Generation with large language models like GPT2 or GPT3
The core algorithm is model agnostic

## TL;DR
### The main project 
It consists out of an algorithm that tries to create metrically correct poetry that also rhymes in german language. It can be tried out by using a google colabs: colabs/poetry_generation. Unfortunately it takes a bit until all packages are installed. In the colab there is also a little description and showcase of the rythm and rhyme algorithms. <br/>
Example: <br/>

(using german-poetry-gpt-2-large)<br/>
Zu tiefst im Menschen , daß sein Antlitz sich nicht <br/>
Von seiner innern Schönheit scheiden möchte ?<br/>
Ein Lächeln , von dem oft in blassen tagen<br/>
Der Schein in blassen Zügen in den Augen<br/>
Der Greise liegt , der in der Seele liegen<br/>
Von sich versteckt , die ihm in Träumen lachen !<br/>
Ein Lächeln ist die Macht , die über die bahnen<br/>
Der Menschen , selbst der Seelen , die in Aeonen<br/>
Nicht weiterreichen , als des Himmels hallen<br/>
Ein Lächeln ja , ein Lächeln ist sie allein ,<br/>
Des Mannes Blick , denn nur von ihm getragen<br/>
Im Leben eines Menschen ist zugegen<br/>
Die Macht , die über die Zeit und über jahrtausend<br/>
Der Ahnen , die gesammt in ganzen Schaaren ,<br/>
An unsern Schultern hangen mit dem gleichen<br/><br/>

There are also some interesting cases which indicate an (not perfect) awarness by the model about the letters inside of the tokens: <br/>

(using german-poetry-gpt-2-large)<br/>
A wie Apfelbaum , der Bäume ohne Kern hat<br/>
B wie Brotfruchtbaum , ein Kern von bloßem Saume ist ;<br/>
C wie Chlorus , der , wenn er im Meer begegnet<br/>
Wenn er erst im Körper war , mit hundert facher Gier<br/>
In den Grund des Wassers kriecht als wie zum lande<br/>
D wie ölbaum , dessen Äste in der Wildnis sind ,<br/>
E wie Eichenbaum , der über die Erde hin eilt<br/>
F wie Fäulnisbaum wenn aus den Büschen fahl und alt<br/>
G wie neuer Kohlkopf , der im Treibhaus noch lebt<br/>
H wie Hülstabaum , ein Baum der noch in Bäumen lebt<br/>
O wie Polfußbaum der sich im Winde wiegt da<br/>
O wie Rosenholz mit seinen gelben Blättern da<br/>
S wie Silberrohr mit dicken Knöpfen drüber<br/>
Aber der Baum der Wald bedeckt die weite Fläche nur<br/>
S wie Seerücken über dem Meer das weit und breit ist <br/>

### Two spin of projects
There are also two smaller projects that reflect on the combination between digital poetry using patterns and large language models<br/>

in colabs/LLM_alliterations you find a colab that creates sentences in which the words begin with certain character patterns. <br/>

Example: <br/>
(german-poetry-gpt2-large):<br/>
der dumpfe Drang des Daseins dich durch das Dasein durch die Dasein dauernd drängt<br/>

(kant-gpt2-large)<br/>
du durchschaust du durch das Dunkel des Denkens dich doch durchs Dasein dieser Dinge dir dein Dasein deutlich
obiectiv seyn ohne Sinn oder Sinn ohne Substanz ohne Seele ohne Schönheit ohne system<br/>

in colabs/llm_the_lutz a colab can be found which creates sentences according to patterns using both, predefined patterns and large language models. It is loosely inspired by Theo Lutz, who programme an Zuse Z22 so that it creates poetry according to patterns form attributed words. In this project a syntax was developed in order to define the patterns. <br/>

Example: <br/>
(kant-gpt2-large)<br/>
du kannst nicht wünschen denn wir dürfen nicht verlangen<br/>
es kann nicht finden denn ich muss suchen<br/>
sie muss vorhergehen denn es darf wirken<br/>

der erste Grad ist die Unlauterkeit denn der der Mensch ist nicht vollkommen<br/>
die Ursache ist nicht die Wirkung denn der Grund ist blos denn die Ursache ist die Folge<br/>
der Mensch ist nicht ein Zweck denn der Zweck mag seyn denn der Mensch ist nicht ein Thier<br/>

er ist ein Recht denn die Gewalt ist eine Verbindlichkeit . er ist nie wirklicher denn der Mensch ist ein Weltwesen<br/>
du bist ein Welttheil denn die Welt ist ein Platz . sie ist ein Begriff denn die Dinge sind nicht<br/><br/>

### Release of finetuned language models
Two models have been trained for this project and are released on huggingface: 
Anjoe/kant-gpt2-large  was trained on the "Akademie Ausgabe" for Kant
Anjoe/german-poetry-gpt2-large was trained on a corpus extracted from projekt gutenberg

### Rhyme and rythm models

Two models have been trained and released: <br/>
sia_rhyme projects words into a 128 dimensional vector space in a way that rhyming words are closer to each other. <br/>
ortho_to_ipa translates words into the IPA phonetic alphabet with symbols for secondary and primary word stress. This enables an algorithm to detect the rythm of the words.<br/>

# The Algorithm

## Rhyme detection
### The corpus

As corpus we use the german rhyme corpus https://github.com/tnhaider/german-rhyme-corpus unfortunately the files that have been annotated are from the Deutsches Textarchiv in the TEI standard. Due to inconsistent usage of the TEI notation there might be noice introduced into the extracted dataset of rhyming and non rhyming word pairs. Additional to this the authors of the dataset also detected some noice due to annotation mistakes. https://www.aclweb.org/anthology/W18-4509/

### unsupervised rhyme detection
A method was introduced and tested in order to detect rhymes in an unsupervised way. Via text to speech (here amazon Polly is used) words are converted to audio files. From these the mfcc features are extracted with the librosa library. By comparing these mfcc features it is possible to detect rhymes. The method is validated on the noicy corpus mentioned above. An accuracy of 93% was detected. <br/>

<img src="graphics/mfccs_verstehen.svg"  title="mfcc features of the german word verstehen">
