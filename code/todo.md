# todo

## hauptsächlich environment bezogen

- [x] TCP- start position im voxel space

## learning bezogen

- [x] reward springt manchmal unerklärbar hoch im Verhältnis zu genommenen steps. why? --> fixed
- [ ] Fehlermeldung Tensorshape mismatch ---> unfortunately still not solved
- [ ] epsilon decay funktioniert nicht. warum?
- [x] Ist das QNetwork das richtige?--> ja (Ari)

## wenn das lernen richtig funktioniert

- [ ] MSE berechnen zwischen idealem und tatsächlichem trajectory
(wir berechnen bereits den MSE zw. den rewards die das netzwerk erwartet und die die tatsächlich erhalten werden, sowie den im nächsten schritt erwarteten reward --> ist das das selbe?)
- [x] Orientation verstehen, fixieren und einbinden (Euler Angles) --> see TCP.ipynb  
- [ ] code aufräumen (z.b. alle prints die nicht fehlermeldungen sind im env auskommentieren oder löschen, etc.)


# nach abgabe am Mittwoch
- [ ] helix_point_list evtl von realen werten [Dennis]
- liste mit allen geringsten distanzen wegen mse pro episode 
- liste mit allen mse über alle episoden on das ploten mit episdoe auf x achse und mse auf y achse