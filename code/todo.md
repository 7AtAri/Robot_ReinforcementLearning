# todo

## environment bezogen

- [ ] Quellenangaben (mit links) für die Formeln
- [ ] gymnasium.spaces. Box Objekt: Doku lesen, printen möglich?, wird das Objekt richtig weiterverarbeitet?
- [ ] brauchen wir den voxel_space dann überhaupt noch?
- [ ] joint angles vor updaten abfragen (max-min grenze) [-180° , + 180°]
- [ ] check correct out size of reset() for input in Qnet
- [ ] process_action() vgl mit main branch version

- [ ] Startkoordinaten fixen auf Helix
- [ ] joint angles + TCP- start position
- [ ] Orientation verstehen und einbinden (Euler Angles) --> see TCP.ipynb
- [ ] Inverse Kinematics?

- [ ] step funktion (vgl. dev_branch mit main)
- [ ] check is_on_helix() funktion wenn startkoordinaten stimmen
- [ ] check reset() bzgl. states.flatten()
- [ ] render() checken nach dem startpos fixed

- [ ] Environment 100% überprüfen

## learning bezogen

sofort korrektur möglich:

- [ ] learning funktion zustandsvektor abfrage löschen?
- [ ]  im main loop:  wenn terminated, macht es sinn noch zu lernen oder lieber if abfrage oder break an der stelle

nach env 100%:

- [ ] überprüfen ob env correct geladen wird mit gymnasium
- [ ] Fehlermeldung QNet checken
