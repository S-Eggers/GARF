# Garf-master

Für das Experiment müssen 4 Datensätze in der Datenbank vorhanden sein, wobei Hosp_rules als Beispiel dient (path_pos = Hosp_rules_copy in config.ini, ändern Sie hier, wenn Sie den Datensatz ändern)  
Hosp_rules ist der ursprüngliche saubere Datensatz  
Hosp_rules_copy ist die Version mit den hinzugefügten Fehlerdaten und ist der Datensatz, den wir reparieren wollen, zunächst leer, mit dem kopierten Blueprint und den über insert_error.py hinzugefügten Fehlern  
Hosp_rules_copy1 sind die Fehlerdaten, die beim Hinzufügen der Fehlerdaten separat entnommen werden, anfangs leer  
Hosp_rules_copy2 sind die eigentlichen korrekten Daten, die den hinzugefügten Fehlerdaten entsprechen, anfangs leer.  

Achtung!  
Der Datensatz Hosp_rules ist nicht in den Erkennungs- und Reparaturprozess involviert, er wird nur als Datenvorlage für die Auswertung der Ergebnisse verwendet, und der entsprechende Code ist nur für insert_error.py, reset.py und eva.py gültig  
Die Datensätze Hosp_rules_copy1 und Hosp_rules_copy2 werden nur bei der Fehlergenerierung zu Vergleichszwecken erzeugt und sind für das Programm nicht relevant, wenn sie nicht benötigt werden, löschen Sie "path2" und "path3" aus insert_error.py. "path3" in insert_error.py.  
Der von Ihnen hinzugefügte Datensatz muss in der letzten Spalte eine Label-Spalte haben, die jedoch leer gelassen werden kann und nur in eva.py bei der Auswertung der Ergebnisse verwendet wird, aber da der Codeprozess die Beseitigung der Auswirkungen der Label-Spalte beinhaltet, wird das Fehlen dieser Spalte die Ergebnisse beeinflussen oder einen Fehler melden  

Dieser Code wurde modularisiert und aufgeteilt, die Standard-Einweg-Training und speichern Sie die Modell-Ergebnisse, wenn tatsächlich verwendet werden, führen Sie bitte mindestens einmal in die Vorwärtsrichtung und einmal in die umgekehrte Richtung, mehrere Läufe können die Leistung Ergebnisse ein wenig verbessern  
In main.py, order = 1 für vorwärts; order = 0 für rückwärts, fügen Sie die Fehlerdaten im zweiten Lauf nicht erneut hinzu, kommentieren Sie bitte insert_error(path_ori, path, error_rate) aus  
insert_error.py wird verwendet, um Fehler hinzuzufügen, es gibt 3 Arten von Fehlern: Rechtschreibfehler, fehlende Daten, zufällige Ersetzung anderer Werte unter derselben Attributspalte, falls nicht benötigt, kommentieren Sie insert_error(path_ori, path, error_rate) ebenfalls aus.  
Das Erhöhen der Werte von g_pre_epochs und d_pre_epochs (d.h. die Anzahl der Iterationen des Modellgenerators und des Diskriminators) in config.ini wird die Leistung um einen kleinen Betrag verbessern, aber mit einem erhöhten Zeitaufwand.  

Erwartete Ergebnisse.  
Testdatensatz mit 10k Datenelementen, Hosp-Datensatz ergibt eine Genauigkeit von 98% ± 1% und eine Wiedererkennung von 65% ± 3%; Lebensmitteldatensatz ergibt eine Genauigkeit von 97% ± 2% und eine Wiedererkennung von 62% ± 5%  
Mit zunehmender Datenmenge verbessert sich die Leistung des Modells, mit 100k für den Hosp-Datensatz und 200k für den Lebensmitteldatensatz in der Arbeit  
Für zusätzliche Daten klicken Sie bitte auf den folgenden Link.

Hosp：https://data.medicare.gov/data/physician-compare  
Food：https://data.cityofchicago.org  
Flight：http://lunadong.com/fusionDataSets.htm  
UIS：https://www.cs.utexas.edu/users/ml/riddle/data.html  




