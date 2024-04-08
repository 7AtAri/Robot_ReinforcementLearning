# todo

## hauptsächlich environment bezogen

- [ ] check if new tcp position outside of

## wenn das lernen prinzipiell funktioniert

1) Hyper-Parameter-Tuning:

    - [ ] guten epsilon decay Wert / Ratio (in Bezug auf num_episodes) finden
    - [ ] gute batch size finden

2) Implementierung abschließen:

    - [ ] schauen ob wirklich was gelernt wird, wenn wir mal 1000 episoden laufen lassen
    - [ ] MSE berechnen zwischen idealem und tatsächlichem trajectory
    (wir berechnen bereits den MSE zw. den rewards die das netzwerk erwartet und die die tatsächlich erhalten werden, sowie den im nächsten schritt erwarteten reward --> ist das das selbe?)
    - [?] Orientation verstehen, fixieren und einbinden (Euler Angles) --> see TCP.ipynb  

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
