# todo

## hauptsächlich environment bezogen

- [ ] TCP Position im env.render() wenn über cnn-learning.py aufgerufen

## wenn das lernen prinzipiell funktioniert

- [ ] schauen ob wirklich was gelernt wird, wenn wir mal 1000 episoden laufen lassen
- [ ] MSE berechnen zwischen idealem und tatsächlichem trajectory
(wir berechnen bereits den MSE zw. den rewards die das netzwerk erwartet und die die tatsächlich erhalten werden, sowie den im nächsten schritt erwarteten reward --> ist das das selbe?)
- [ ] Orientation verstehen, fixieren und einbinden (Euler Angles) --> see TCP.ipynb  
- [ ] code aufräumen (z.b. alle prints die nicht fehlermeldungen sind im env auskommentieren oder löschen, etc.)

## learning bezogen abgehakt

- [x] reward springt manchmal unerklärbar hoch im Verhältnis zu genommenen steps. why?
- [x] Fehlermeldung Tensorshape mismatch ---> solved
- [x] epsilon decay funktioniert wieder
- [x] Ist das QNetwork das richtige?--> ja
