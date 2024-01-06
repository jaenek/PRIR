Wejściowe bitmapy znajdują się w in<indeks rozmiaru obrazu>.pgm.
Wyjścia uruchomień programu dla poszczególnych funkcji implementujących paralelizacje
znajdują się w plikach *_out.
Przetworzone bitmapy znajdują się w out<indeks rozmiaru obrazu>_<indeks rozmiaru jądra>.pgm.
Najlepiej działanie filtra widać na mniejszych obrazach.

Skrypt plot.py tworzy funkcję z danych data.csv skopiowanych z wyjścia działania programu.
W pliku plot.png znajduje się obraz z czasem działania poszczególnych funkcji paralelizujących
dla obrazów o rozmiarach: 256x256, 1024x1024, 4096x4096 oraz jąder: 2x2, 3x3, 5x5.

Najlepiej działa paralelizacja w funkcji conv_p2 czyli tylko i wyłącznie po zewnętrzu
zagnieżdżonych pętli iterujących po każdym pikselu obrazu.
Jednak funkcja conv_p3 czyli paralelizacja obu pętli może przyśpieszyć dla większych obrazów
ponieważ wyniki tych dwóch funkcji są bardzo zbliżone.
