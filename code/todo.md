# todo

## gif von tcp auf helix (Dennis)

- [ ] render() speichen alle 5-10 steps
- [ ] in Ordner mit eigenem episode name
- [ ] lange episoden ordner behalten, kurze löschen (zähle elemente in Ordner)
- [ ] MSE plot speichern in selben Ordner statt show()

## wenn das lernen prinzipiell funktioniert

1) Hyper-Parameter-Tuning:

    - [ ] guten epsilon decay Wert / Ratio (in Bezug auf num_episodes) finden
    - [ ] gute batch size finden

2) Implementierung abschließen:

    - [ ] schauen ob wirklich was gelernt wird, wenn wir mal 1000 episoden laufen lassen
    - [ ] Orientation in Network einbinden (Ari)

## allgemeines

- [ ] code aufräumen (z.b. alle prints die nicht fehlermeldungen sind im env auskommentieren oder löschen, etc.)
- [ ] Doku für den Code
- [ ] nebenher die Präsentation anfangen zu machen

## learning bezogen abgehakt

- [x] reward springt manchmal unerklärbar hoch im Verhältnis zu genommenen steps. why?
- [x] Fehlermeldung Tensorshape mismatch ---> solved
- [x] epsilon decay funktioniert wieder
- [x] Ist das QNetwork das richtige?--> ja
- [x] TCP Position im env.render() wenn über cnn-learning.py aufgerufen
