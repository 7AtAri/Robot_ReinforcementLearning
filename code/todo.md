# todo

## hauptsächlich environment bezogen

- [x] TCP- start position im voxel space

## learning bezogen

- [ ] reward springt manchmal unerklärbar hoch im Verhältnis zu genommenen steps. why?
- [ ] Fehlermeldung Tensorshape mismatch ---> remember() checken!
- [ ] epsilon decay funktioniert nicht. warum?
- [ ] Ist das QNetwork das richtige? (Ari)

## wenn das lernen richtig funktioniert

- [ ] MSE berechnen zwischen idealem und tatsächlichem trajectory
(wir berechnen bereits den MSE zw. den rewards die das netzwerk erwartet und die die tatsächlich erhalten werden, sowie den im nächsten schritt erwarteten reward --> ist das das selbe?)
- [ ] Orientation verstehen, fixieren und einbinden (Euler Angles) --> see TCP.ipynb  
- [ ] code aufräumen (z.b. alle prints die nicht fehlermeldungen sind im env auskommentieren oder löschen, etc.)

## eventuelle alternative

- [ ] anderen learning algorithmus (einen von denen aus dem Unterricht auf unser Problem angepasst) mit unserem environment implementieren?
