# Methode und Reproduzierbarkeit

## Fragestellung und Daten

Untersucht wird, welche Anliegen in einer zufälligen Auswahl historischer
Finanzbeschwerden besonders häufig vorkommen. Die Datenquelle ist die von Selene
Reyes am 13. Mai 2019 bereitgestellte Version 1 der Consumer Complaint Database,
ursprünglich vom Consumer Financial Protection Bureau. `analyse.py prepare`
prüft die SHA256 des ZIP-Archivs vor jedem Import. Die Datei enthält 18 Spalten;
für die Verarbeitung werden nur ID, Datum, Produkt und Beschwerdetext eingelesen.
Nur Text wird zur Modellierung verwendet. Das Produktfeld wird mitgeführt,
aber weder für Vektorisierung, Training noch Themenzahlauswahl verwendet.

Die Beschwerdetexte werden bereits in der Quelle teilweise durch X-Marker
maskiert. Fehlende Texte, normalisierte Dubletten und zusätzliche identische
Lemmasequenzen werden entfernt. Die Normalisierung löscht auch Zahlen: Dadurch
können Beschwerden, die sich nur in Beträgen oder Daten unterscheiden,
zusammenfallen. Das wird als bewusste Einschränkung dokumentiert.
Spracherkennung mit langdetect dient als Filter, nicht als garantierter Nachweis
der Sprache jedes Textes (Danilak, n.d., Abschnitt Languages und Basic usage).
Der Zufallsstartwert wird festgesetzt.

spaCy segmentiert und lemmatisiert. Personen und Orte erkannte die kleine
englische Pipeline nur heuristisch; ihre Tokens werden ausgeschlossen.
Organisationen werden beibehalten, weshalb Markennamen weiterhin Themen prägen
können. Verneinungen bleiben als Wörter erhalten; ein Bag-of-Words-Modell bildet
trotzdem weder Wortreihenfolge noch den Geltungsbereich einer Verneinung ab.

## Vektorisierung

CountVectorizer zählt Unigramme. TfidfTransformer gewichtet dieselben Einträge
mit `idf(t) = log((1+n_train)/(1+df_train(t))) + 1` und normiert anschließend jede
Dokumentzeile auf L2-Norm 1. Vokabular und IDF werden ausschließlich aus 4.000
Trainingsdokumenten geschätzt. Dieselben Regeln transformieren die 1.000
Validierungsdokumente. Beide Darstellungen besitzen gleiche Spalten und dieselbe
Besetzungsstruktur, unterscheiden sich aber in ihren Gewichten (scikit-learn
developers, n.d.-b, Kap. 7.2.3).

Zum Vergleich werden Dimension, Besetzungsdichte und Speicherbedarf berichtet.
An 300 zufällig ausgewählten Trainingsdokumenten werden außerdem alle 44.850
ungeordneten Dokumentpaare per Kosinusähnlichkeit verglichen. Die Übereinstimmung
des jeweils nächsten Nachbarn bezieht sich auf diesen gemeinsamen 300er-Pool.
Eine zusätzliche Diagnose zählt Paare mit TF-IDF-Kosinus >= 0,95 im gesamten
5.000er-Korpus, einschließlich Paaren über die Trainings-/Validierungsgrenze.

## Themenmodelle und Auswahl

LDA modelliert diskrete Wörter über Themenmischungen. NMF zerlegt die TF-IDF-
Matrix additiv in nichtnegative Dokument- und Themenfaktoren (scikit-learn
developers, n.d.-a, Kap. 2.5.7-2.5.8). Bei LDA werden Batch-Lernen, maximal 40
Iterationen, doc_topic_prior=0,1 und topic_word_prior=0,01 verwendet. Die
Perplexitätsänderung kann ab einer Prüfung alle fünf Iterationen unter 0,1 zur
frühen Beendigung führen. NMF verwendet Coordinate Descent, zufällige
Initialisierung, maximal 800 Iterationen und Toleranz 0,0001. Konvergenzwarnungen
werden gespeichert und vor der Freigabe geprüft. Identische Zufallswerte über
verschiedene Verfahren erzwingen keine inhaltlich identischen Initialisierungen.

In der ausgeführten Untersuchung erreichten alle LDA-Fits die Obergrenze von
40 Iterationen. Es trat keine Konvergenzwarnung auf; das belegt dennoch keine
vollständige Konvergenz. Ein höheres Iterationsbudget kann die Ergebnisse ändern.

Der vor dem Training festgehaltene [Versuchsplan](versuchsplan.md) beschreibt
Startgitter, Verfeinerung und Auswahlregeln. Hauptmaß ist c_v nach Röder et al.
(2015), implementiert in Gensim. Verwendet werden stets zehn Topwörter und
dieselben Validierungstexte mit Fenstergröße 110. Die Statistik wird einmal für
alle Themenwortlisten aufgebaut. c_v ist eine auf Wort-Kookkurrenzen beruhende
Näherung an Zusammenhang, kein Beweis sachlicher Richtigkeit.

Fehlt ein Topwort vollständig im Validierungsreferenzkorpus, kann c_v undefiniert
sein. Solche Werte werden nicht durch 0 ersetzt und nicht stillschweigend beim
Mittelwert ausgelassen. Der betroffene k-Kandidat erhält einen fehlenden
Kohärenzmittelwert und scheidet aus der metrischen Vorauswahl aus. Die CSV lässt
die betreffende Zelle leer; `coherence_diagnostics.json` nennt betroffene Modelle
und Wörter. Andere Diagnosewerte bleiben verfügbar.

Für jedes k werden drei Seeds verglichen. Thema 1 eines Durchlaufs muss nicht
Thema 1 eines anderen entsprechen. Daher maximiert eine Eins-zu-eins-Zuordnung
die Kosinusähnlichkeit der Themenwortvektoren. Der Durchschnitt über zugeordnete
Themen und die drei Seed-Paare ist das Stabilitätsmaß. Zusätzlich wird der
permutationsinvariante Adjusted Rand Index dominanter Validierungszuordnungen
berichtet. Diese Stabilität beschreibt Initialisierungssensitivität, keine
Stabilität gegenüber einer neuen Stichprobe.

Mittelwert +/- Standardabweichung über drei Seeds ist eine deskriptive Streuung,
kein Konfidenzintervall. LDA-Perplexität und NMF-Rekonstruktionsfehler sind nur
innerhalb desselben Verfahrens und derselben Repräsentation vergleichbar.
Metrisch gute Vorhersage kann von menschlicher Interpretierbarkeit abweichen
(Chang et al., 2009, S. 1). Die endgültige Entscheidung kombiniert deshalb
Kohärenz, Stabilität, Redundanz, Themenanteile und eine inhaltliche Textprüfung.

## Häufigkeiten und Interpretationsgrenzen

Der repräsentative Seed maximiert die mittlere Themenähnlichkeit zu den anderen
Seeds (Medoid). NMF besitzt eine Skalierungsmehrdeutigkeit: Eine beliebige
Multiplikation einer Themenzeile und inverse Skalierung der zugehörigen
Dokumentspalte lässt die Rekonstruktion unverändert. Vor der Dominanzzuordnung
wird deshalb jede Themenzeile auf L1-Norm 1 gebracht und die Dokumentspalte
entsprechend angepasst. Anschließend werden Dokumentgewichte auf Summe 1
normiert. Die so erhaltenen NMF-Anteile sind keine kalibrierten Wahrscheinlichkeiten.

Die Rangliste zählt jedes Dokument genau einmal nach seinem größten
Themengewicht. Mehrere Anliegen können trotzdem im selben Dokument vorkommen.
Anteile über Trainings- und Validierungsdokumente beschreiben die analysierte
5.000er-Stichprobe. Sie erlauben keine Aussagen über heutige Beschwerden,
Kommunen, repräsentative Bevölkerungsanteile oder Dringlichkeit.
Eine unüberwachte Themenanalyse liefert auch keinen Nachweis, dass die in
Beschwerden behaupteten Sachverhalte zutreffen.

## Ausführungsumgebung

Python 3.12, isolierte venv, direkte Pins in `requirements.txt`, vollständiger
Abhängigkeitsexport in `requirements-lock.txt`, konkrete Laufumgebung in
`results/environment.json`. Alle Analysen laufen lokal. Die Kennzahlen stammen
aus den ausgeführten Skripten. Themenlabels und Begründungen sind in
`interpretation.json` dokumentiert. Die qualitative Prüfung basiert auf wenigen
Textausschnitten und ersetzt keine systematische Bewertung durch mehrere Personen.

## Quellen

Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C. & Blei, D. M. (2009).
Reading tea leaves: How humans interpret topic models. In Advances in neural
information processing systems 22. https://www.cs.columbia.edu/~blei/papers/ChangBoyd-GraberWangGerrishBlei2009a.pdf

Danilak, M. (n.d.). langdetect. GitHub. https://github.com/Mimino666/langdetect

Explosion. (n.d.). en_core_web_sm (Version 3.8.0) [Sprachmodell].
https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0

Gensim developers. (n.d.). models.coherencemodel: Topic coherence pipeline.
https://radimrehurek.com/gensim/models/coherencemodel.html

Reyes, S. (13. Mai 2019). Consumer complaint database (Version 1) [Datensatz].
Kaggle. https://www.kaggle.com/datasets/selener/consumer-complaint-database/versions/1

Röder, M., Both, A. & Hinneburg, A. (2015). Exploring the space of topic coherence
measures. In Proceedings of the eighth ACM international conference on web
search and data mining (S. 399-408). ACM. https://doi.org/10.1145/2684822.2685324

scikit-learn developers. (n.d.-a). Decomposing signals in components (matrix
factorization problems). scikit-learn 1.8.0 documentation.
https://scikit-learn.org/1.8/modules/decomposition.html

scikit-learn developers. (n.d.-b). Feature extraction. scikit-learn 1.8.0
documentation. https://scikit-learn.org/1.8/modules/feature_extraction.html
