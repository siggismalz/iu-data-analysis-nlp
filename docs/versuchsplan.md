# Versuchsplan vor der Modellschätzung

Stand: 07.10.2026. Ziel ist eine begründete Themenauflösung für historische
Verbraucherbeschwerden. Die Auswahl gilt für diesen Datenstand, diese Stichprobe
und die dokumentierte Vorverarbeitung, nicht als universell wahre Themenanzahl.

## Daten und Vergleich

Kaggle-Kopie von Selene Reyes, Version 1 (13.05.2019), Quelle CFPB.
Leere Texte und normalisierte Textdubletten werden vor der Stichprobe entfernt.
Aus einer mit Seed 42 permutierten Reihenfolge werden die ersten 5.000 geeigneten
Texte genommen: Spracherkennung Englisch mit Wahrscheinlichkeit >= 0,9, mindestens
15 verwertbare Tokens nach Bereinigung. Dieses Verfahren entspricht einer
Zufallsstichprobe aus den nach diesen Regeln geeigneten Texten, ohne das gesamte
Archiv sprachlich und linguistisch verarbeiten zu müssen.

URLs, E-Mail-Adressen, X-Maskierungen und Zahlen werden entfernt; spaCy 3.8 mit
en_core_web_sm 3.8.0 lemmatisiert. Stoppwörter werden entfernt, no/not/never/nor
bleiben. Personen- und Ortsentitäten werden ausgeschlossen. Das ist eine
heuristische Datenminimierung, keine garantierte Anonymisierung. Rohtexte und
Dokumentzuordnungen bleiben lokal. Metadaten wie Product/Issue sind keine Features.
Nahe Dubletten werden in der Stichprobe diagnostiziert; verbleibende Vorlagentexte
werden als Einschränkung diskutiert.

80 % Training, 20 % Validierung, zufällig mit Seed 42. CountVectorizer lernt nur
auf Training ein gemeinsames Unigramm-Vokabular (min_df=5, max_df=0,8, höchstens
6.000 Wörter). TfidfTransformer lernt IDF nur auf Training. LDA erhält Counts,
NMF TF-IDF. Gleiche Dokumente und Spalten erlauben einen direkten Vergleich der
Vektordarstellungen. Ein Vergleich beider Pipelines isoliert den Einfluss der
Vektorisierung jedoch nicht vom Einfluss des Themenmodells.

## Modellwahl

Startgitter k = 5, 8, 10, 12, 15, 20; je drei Seeds 17, 42, 73. NMF verwendet
echte Zufallsinitialisierung. Bei LDA bleiben Priors über k konstant. Es werden
keine Kategorien als Zielvariable genutzt.

1. Gemeinsames Hauptmaß: c_v der zehn wichtigsten Wörter, auf denselben
   Validierungstexten, Fenster 110. Mittelwert und Standardabweichung über Seeds.
2. Stabilität: paarweise Kosinusähnlichkeit der Themenwortvektoren, optimale
   Eins-zu-eins-Zuordnung (Hungarian-Algorithmus), Durchschnitt über Themen und
   Seed-Paare. Ergänzend ARI dominanter Dokumentzuordnungen.
3. Redundanz: maximale Kosinusähnlichkeit zwischen Themen desselben Modells;
   Vielfalt: Anteil unterschiedlicher Wörter unter allen Top-10-Wörtern.
4. Zusatzdiagnostik: LDA-Perplexität und relativer NMF-Rekonstruktionsfehler auf
   Validierung. Diese Größen werden nicht zwischen LDA und NMF verrechnet.
5. Vorauswahl pro Verfahren: c_v höchstens 0,02 unter dem besten Mittelwert,
   Stabilität mindestens 0,80, jedes Thema bei mindestens 1 % der
   Validierungsdokumente dominant, maximale Themenähnlichkeit höchstens 0,90.
   Unter geeigneten Kandidaten wird das kleinste k vorgeschlagen. Diese
   Schwellen sind praktische, vorab festgelegte Heuristiken, keine Naturgesetze.
   Falls kein Kandidat alle Kriterien erfüllt, wird das ausdrücklich dokumentiert
   und nach Kohärenz/Stabilität plus inhaltlicher Prüfung entschieden.
6. Verfeinerung: unmittelbare Nachbarn des vorgeschlagenen k und des
   Kohärenzmaximums. Liegt das Maximum bei 20, zusätzlich 25 und 30; ein weiter
   steigendes Ergebnis wird als offene Grenze der Suche ausgewiesen.
7. Inhaltliche Prüfung der Vorschläge und Nachbarn anhand von Topwörtern,
   drei starken Beispielen und zwei zufällig gewählten Texten pro Thema.
   Themenlabels und Inhaltsnotizen werden zusammen mit ihrer Begründung
   dokumentiert. Die Stichprobensichtung ersetzt keine systematische Bewertung
   durch mehrere Personen.

Die Dokumentanteile werden für alle 5.000 Texte mit den auf Training geschätzten
Modellen bestimmt. Es erfolgt keine Neuschätzung auf der Validierungsmenge nach
Auswahl; Validierungswerte sind trotzdem Auswahlwerte, keine unabhängigen Testwerte.
Ein repräsentativer Seed wird als Themen-Medoid (höchste mittlere Stabilität zu
den anderen Seeds) gewählt. NMF-Faktoren werden vor Dominanzzuordnung so skaliert,
dass jede Themenwortzeile L1-Norm 1 hat; die Dokumentfaktoren werden entsprechend
gegengleich skaliert. Ihre normierten Anteile sind keine LDA-Posteriorwahrscheinlichkeiten.

Quellen und genaue technische Definitionen folgen in der Methodendokumentation.

Präzisierung während der Verfeinerung: Verschiebt sich das Kohärenzmaximum,
werden auch dessen unmittelbare noch fehlende Nachbarn ergänzt. Maximal drei
solche Nachbarschaftsschritte, begrenzt auf k=2..30. Die numerischen Auswahlregeln
bleiben unverändert; eine dadurch noch offene Suchgrenze wird berichtet.
