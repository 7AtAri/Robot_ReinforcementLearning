# todo

## hauptsächlich environment bezogen

Fehler die gefixt werden müssen:

- [ ] reset() und step() übergeben nur einen state aus dem observation_space
- [ ] do we need a close() method ? --> Closes the environment, important when external software is used, i.e. pygame for rendering (matplotlib?)

Dennis:

- [x] Quellenangaben (mit links) für die Formeln
- [x] joint angles vor updaten abfragen (max-min grenze) [-180°, + 180°]
- [x] process_action() vgl mit main branch version

Ari: (ich hab mir mal die zwei box object todos geholt, ich glaub das macht mehr sinn wenn ich die fixe,
weil die direkt ins QNet gehen.)

- [x] Startkoordinaten fixen auf Helix
- [x] joint angles + TCP- start position
- [x] Inverse Kinematics needed?  --> wird glaub ich nicht benötigt, außer 1 mal am Anfang für die Startposition  --> extrafile
- [ ] Orientation verstehen und einbinden (Euler Angles) --> see TCP.ipynb  --> ist glaub ich nicht nötig...und aktuell auch nicht so wichtig

Gemeinsam:

- [ ] gymnasium.spaces. Box Objekt: Doku lesen, printen möglich?, wird das Objekt richtig weiterverarbeitet?
- [ ] brauchen wir den voxel_space dann überhaupt noch?
- [ ] check reset() bzgl. states.flatten()
- [ ] check correct out size of reset() for input in Qnet

Gemeinsam, wenn das Env steht:

- [ ] step funktion (vgl. dev_branch mit main)
- [x] check is_on_helix() funktion wenn startkoordinaten stimmen   -->  correclty returns True now!
- [x] render() checken nach dem startpos fixed

- [ ] Environment 100% überprüfen

## learning bezogen

---> @HUmasterFF: das was hier schon geht, kannst du gern schon mal machen

sofort korrektur möglich:

- [ ] learning funktion zustandsvektor abfrage löschen?
- [ ]  im main loop:  wenn terminated, macht es sinn noch zu lernen oder lieber if abfrage oder break an der stelle

nach env 100%:

- [ ] überprüfen ob env correct geladen wird mit gymnasium
- [ ] Fehlermeldung QNet checken
