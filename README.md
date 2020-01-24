# Projekt semantycznej segmentacji 

##  Jest to rozwiązanie do konkursu kaggla, w którym zadaniem był model segmentacji różnego typu chmur.

## Wykorzystany model to Unet, zapożyczony z tej strony:
### https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

## 1. Preprocessing danych
### Pocięcie pliku jpeg o wysokiej rozdzielczości na mniejsze(w tym wypadku 140x140).
### Wygenerowanie masek z pliku csv jako tzw. True label maski (w tym wypadku 4 kanałowy obrazek dla 4ch klas).

## 2. Testowanie
### Plik combine_images.py testuje czy funkcjonalność krojenia obrazka wraz z maską przebiega pomyślnie.

## 3. Przyjęte miary (losses.ipynb).
### Jako miarę trenowania modelu zastosowałem loss =  SoftDiceLosss(X, y) + BCE(X, y)
### Wynik optymalizacji modelu można znaleźć w pliku losses.ipynb .

## 4. Dane
### a) Pobrać dane ze strony konkursu kaggla
### b) Rozpakować dane do katalogu data/ (na tym samym poziomie co src albo README.md).
### c) Uruchomić `python gen_data.py` w katalogu src. To polecenie pokroi obrazki wraz z plikiem csv masek do nowych plików.


## Trenowanie modelu
### a) Pobrać i wykenerować małe obrazki jak w pkt 4.
### c) Uruchomić `python train_data.py`. Domyślnie zostanie wykonane 10 epok.


## Predykcja modelu
### Istnieje możliwość wygenerowania predykcji na wstępnie wytrenowanym modelu. W tym celu należy:
### a) Pobrać obrazki i rozpakować je w katalogu data/ , jak pokazano w pkt.4 .
###  Do predykcji nie wymagane jest polecenie `python gen_data.py`,  ponieważ plik `predict.ipynb` dokonuje predykcji na pełnowymiarowych obrazkach.

