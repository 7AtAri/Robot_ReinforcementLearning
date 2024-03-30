# todo

## hauptsächlich environment bezogen

Fehler die gefixt werden müssen:

- [ ] reset() und step() übergeben den current state aus dem observation_space (den voxel_space und die TCP position?)

Dennis:

- [x] Quellenangaben (mit links) für die Formeln
- [x] process_action() vgl mit main branch version
- [ ] joint angles vor updaten abfragen (max-min grenze) [-180°, + 180°] ---> pro joint abfragen

Ari:

- [x] Startkoordinaten fixen auf Helix
- [x] joint angles + TCP- start position
- [x] Inverse Kinematics needed?  --> wird glaub ich nicht benötigt, außer 1 mal am Anfang für die Startposition  --> extrafile

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
- [ ] im main loop:  wenn terminated, macht es sinn noch zu lernen oder lieber if abfrage oder break an der stelle

nach env 100%:

- [ ] überprüfen ob env correct geladen wird mit gymnasium
- [ ] Fehlermeldung QNet checken

## wenn das lernen funktioniert

- [ ] render() im env von plt.show() auf plt.save(...) umstellen
- [ ] Orientation verstehen und einbinden (Euler Angles) --> see TCP.ipynb  --> ist glaub ich nicht nötig...und aktuell auch nicht so wichtig
- [ ] alle prints die nicht fehlermeldungen sind im env auskommentieren oder löschen

- [ ] besseres Network: DQN mit 2 channels(voxel_space und tcp_pos in voxel space translated) conv3D() und lin layers als output
- [ ] find out where the .push() method is used?
- [ ] implement epsilon decay instead of epsilon greedy
