######################################
jambus = [0,1]                        # defining the metric patterns
trochee = [1,0]
######################################

#Joseph Karl Benedikt, Freiherr von Eichendorff  Laue Luft kommt blau geflossen 
prompt_1 = ['''Laue Luft kommt blau geflossen,
Frühling, Frühling soll es sein!
Waldwärts Hörnerklang geschossen,
Mutger Augen lichter Schein;
Und das Wirren bunt und bunter
''',[10],trochee]                                     # a prompt is a list of a text followed by the syllable list and the meter

# Friedrich Schiller Die Künstler
prompt_2 = ['''Nur durch das Morgentor des Schönen
Drangst du in der Erkenntnis Land.
An höhern Glanz sich zu gewöhnen,
Übt sich am Reize der Verstand.
''',[9,8],jambus]

# Friedrich Schiller Die Künstler
prompt_3 = ['''Als der Erschaffende von seinem Angesichte
Den Menschen in die Sterblichkeit verwies
Und eine späte Wiederkehr zum Lichte
Auf schwerem Sinnenpfad ihn finden hieß,
''',[11],jambus]

# Friedrich Schiller Genialität
prompt_4 = ['''Wodurch gibt sich der Genius kund? Wodurch sich der Schöpfer
Kund gibt in der Natur, in dem unendlichen All:
Klar ist der Äther und doch von unermeßlicher Tiefe;
Offen dem Aug, dem Verstand bleibt er doch ewig geheim.
''',[12,14],trochee]

# Johannes Daniel Falk An das Nichts
prompt_5 = ['''Selbst philosophische Systeme –
Kants Lieblingsjünger, Reinhold, spricht’s –
Von Plato bis auf Jakob Böhme,
Sie waren samt und sonders – Nichts.

Was bin ich selbst? – Ein Kind der Erde,
Der Schatten eines Traumgesichts,
Der halbe Weg von Gott zum Werde,
Ein Engel heut, und morgen – Nichts.
''',[9],jambus]

# Johann Wolfgang von Goethe Vermächtnis
prompt_6 = ['''Kein Wesen kann zu nichts zerfallen!
Das Ewge regt sich fort in allen,
Am Sein erhalte dich beglückt!
Das Sein ist ewig: denn Gesetze
Bewahren die lebendgen Schätze,
Aus welchen sich das All geschmückt.
''',[9,11],jambus]

# Johann Wolfgang von Goethe Parabase
prompt_7 = ['''Freudig war, vor vielen Jahren,
Eifrig so der Geist bestrebt,
Zu erforschen, zu erfahren,
Wie Natur im Schaffen lebt.
Und es ist das ewig Eine,
Das sich vielfach offenbart
''',[8],trochee]

# Giacomo Graf Leopardi Palinodie an den Marchese Gino Capponi
prompt_8 = ['''O Geist, o Einsicht, Scharfsinn, übermenschlich,
Der Zeit, in der wir leben! Welches sichre
Philosophiren, welche Weisheit lehrt
In den geheimsten, höchsten, feinsten Dingen
Den kommenden Jahrhunderten das unsre!
Mit welcher ärmlichen Beständigkeit
Wirft heut der Mensch vor das, was gestern er
Verspottet, sich auf's Knie, um morgen wieder
Es zu zertrümmern, dann aufs neu die Trümmer
Zu sammeln, es auf den Altar zurück
Zu setzen, es mit Weihrauch zu bequalmen!
''',[10,12],jambus]

# Friedrich Hebbel Philosophenschicksal
prompt_9 = ['''Salomons Schlüssel glaubst du zu fassen und Himmel und Erde
Aufzuschließen, da löst er in Figuren sich auf,
Und du siehst mit Entsetzen das Alphabet sich erneuern,
Tröste dich aber, es hat währende der Zeit sich erhöht.
''',[12],trochee]

# Heinrich Heine Himmelfahrt
prompt_10 = ['''Die Philosophie ist ein schlechtes Metier.
 Wahrhaftig, ich begreife nie,
Warum man treibt Philosophie.
Sie ist langweilig und bringt nichts ein,
Und gottlos ist sie obendrein;
''',[11],jambus]

# Friedrich Schiller Jeremiade
prompt_11 = ['''Alles in Deutschland hat sich in Prosa und Versen verschlimmert,
Ach, und hinter uns liegt weit schon die goldene Zeit!
Philosophen verderben die Sprache, Poeten die Logik.
''',[12,13],trochee]

# Robert Gernhardt, Trost und Rat
prompt_12 = ['''Ja wer wird denn gleich verzweifeln,
weil er klein und laut und dumm ist?
Jedes Leben endet. Leb so,
daß du, wenn dein Leben um ist

von dir sagen kannst: Na wenn schon!
Ist mein Leben jetzt auch um,
habe ich doch was geleistet:
ich war klein und laut und dumm.
''',[8,10],trochee]

# Robert Gernhardt, Ach!
prompt_13 =['''Woran soll es gehn? Ans Sterben?
Hab ich zwar noch nie gemacht,
doch wir werd’n das Kind schon schaukeln —
na, das wäre ja gelacht!

Interessant so eine Sanduhr!
Ja, die halt ich gern mal fest.
Ach – und das ist Ihre Sense?
Und die gibt mir dann den Rest?
''',[10,8],trochee]

# Antonio Cho
prompt_14 = ['''Gibt es einen Dadageist?
Ich behaupte, er sei
das große Gelächter
das einzig umfassende metaphysische Gelächter
das große Gelächter
über den Witz der Schöpfung
das große Lachen
über den Witz der eigenen Existenz.
''',[10,9],jambus]

prompt_15 = ['''über den Feldhamster Karl und den Philosophen Kant:
''',[10,8],trochee]

prompts = [prompt_2,prompt_3,prompt_5,prompt_7,prompt_8,prompt_10,prompt_11,prompt_12,prompt_13]