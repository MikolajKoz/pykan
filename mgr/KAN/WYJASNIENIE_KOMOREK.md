# Wyjasnienie komorek notatnika KAN_research.ipynb

Ponizej szczegolowe omowienie kazdej z 45 komorek notatnika badawczego.

---

## Komorka [0] - Markdown: Tytul i wstep

Naglowek pracy: *Analiza porownawcza KAN vs MLP w zadaniach regresji*.
Krotki opis celu badan -- porownanie Kolmogorov-Arnold Networks z klasycznymi MLP
w kontekscie pracy magisterskiej. Podano 5 kluczowych pytan badawczych.

---

## Komorka [1] - Markdown: Sekcja 1 naglowek

Naglowek sekcji *Konfiguracja srodowiska i diagnostyka GPU/CPU*.

---

## Komorka [2] - Kod: Import bibliotek i diagnostyka urzadzenia

**Co robi:** Importuje wszystkie potrzebne biblioteki (numpy, pandas, matplotlib,
seaborn, torch, kan, tensorflow, sklearn) i automatycznie wykrywa dostepne GPU.

**Kluczowe koncepty:**
- `torch.cuda.is_available()` -- sprawdza czy dostepna jest karta NVIDIA z CUDA
- `tf.config.list_physical_devices('GPU')` -- analogicznie dla TensorFlow
- `DEVICE` -- globalna zmienna okreslajaca gdzie beda obliczenia ('cuda' lub 'cpu')
- Import `KAN` z biblioteki `kan` (PyKAN) -- glowna klasa sieci KAN

**Dlaczego:** Uniwersalny kod, ktory dziala zarowno na GPU jak i CPU bez zmian.

---

## Komorka [3] - Markdown: Sekcja 2 naglowek

Naglowek sekcji *Klasy pomocnicze*.

---

## Komorka [4] - Kod: Klasa KANDatasets

**Co robi:** Definiuje klase narzedzowa do generowania zbiorow danych.
Kazdy zbiór zwraca slownik w formacie PyKAN: `{'train_input', 'train_label',
'test_input', 'test_label'}` jako tensory PyTorch na odpowiednim urzadzeniu.

**Metody:**
- `create_synthetic_dataset(func, n_samples, noise, dim)` -- uniwersalny generator
- `create_polynomial_dataset()` -- wielomian $x^3 - 2x^2 + x + 0.5$
- `create_sinusoidal_dataset()` -- funkcja sinusoidalna $\sin(2\pi x)$
- `create_2d_dataset()` -- $e^{\sin(\pi x_1) + x_2^2}$ (2 zmienne)
- `create_diabetes_dataset()` -- prawdziwy zbior danych z sklearn (442 probki, 10 cech)

**Kluczowe koncepty:**
- Format danych PyKAN -- slownik z 4 tensorami (train/test x input/label)
- Tensory sa automatycznie przenoszone na `DEVICE` (GPU/CPU)
- Podzial 80/20 train/test z losowym mieszaniem
- Standaryzacja i normalizacja dla zbioru Diabetes

---

## Komorka [5] - Kod: Klasa KANRegressor

**Co robi:** Wrapper na klase `KAN` z PyKAN, upraszczajacy trening i ewaluacje.

**Parametry konstruktora:**
- `width` -- architektura sieci, np. `[2, 5, 1]` (2 wejscia, 5 neuronow, 1 wyjscie)
- `grid` -- liczba punktow siatki B-spline (G) -- im wiecej, tym dokladniejsze splajny
- `k` -- stopien splajnu B-spline (domyslnie 3 = kubiczny)
- `lamb` -- sila regularyzacji L1
- `reg_metric` -- metryka regularyzacji (np. 'edge_forward_spline_n')

**Metoda `fit()`:**
- Uzywa optymalizatora LBFGS (domyslnie w PyKAN)
- Trenuje przez `steps` krokow
- Zwraca historje strat (train/test loss)

**Metoda `predict()`:** Przepuszcza dane przez siec i zwraca predykcje.

**Metoda `evaluate()`:** Oblicza R2, MSE, RMSE, MAE na zbiorze testowym.

**Kluczowe koncepty:**
- **B-spline:** Funkcje bazowe splajnow -- kazda krawedz w KAN ma uczona funkcje
  aktywacji zdefiniowana jako kombinacja liniowa B-spline'ow
- **LBFGS:** Optymalizator quasi-Newtonowski, dobrze dziala dla KAN bo
  uzywa aproksymacji hessianu (macierzy drugich pochodnych)
- **G (grid):** Liczba interwalow siatki; wiecej = wiecej parametrow i dokladnosc
- **k (degree):** Stopien splajnu; k=3 to kubiczny, k=5 to piatego stopnia

---

## Komorka [6] - Kod: Klasa MLPRegressor (TensorFlow/Keras)

**Co robi:** Implementuje klasyczny MLP za pomoca TensorFlow/Keras.

**Parametry:**
- `hidden_layers` -- lista rozmiarow warstw, np. `[32, 16]`
- `activation` -- funkcja aktywacji (relu, tanh, selu)
- `learning_rate` -- tempo uczenia dla optymalizatora Adam
- `epochs`, `batch_size` -- parametry treningu

**Budowa modelu:** `tf.keras.Sequential` z warstwami Dense, kazda z podana aktywacja.
Ostatnia warstwa ma 1 neuron (regresja) bez aktywacji.

**Kluczowe roznice KAN vs MLP:**
- MLP: stale aktywacje (np. ReLU) na neuronach
- KAN: uczone aktywacje (B-spline) na krawedziach
- MLP: parametry to wagi polaczen + biasy
- KAN: parametry to wspolczynniki B-spline na kazdej krawedzi

---

## Komorka [7] - Markdown: Sekcja 3 naglowek

Naglowek *Liczba parametrow* z wzorami:
- **KAN:** $P = \sum n_l \cdot n_{l+1} \cdot (G+k+1)$ na warstwe
- **MLP:** $P = \sum n_l \cdot n_{l+1} + n_{l+1}$ na warstwe

---

## Komorka [8] - Kod: Liczenie parametrow

**Co robi:** Implementuje wzory na liczbe parametrow i tworzy tabele porownawcza.

**KAN:** Kazda krawedz ma $G + k + 1$ parametrow (wspolczynniki B-spline):
- $G$ interwalow siatki
- $k$ stopien splajnu
- $+1$ dodatkowy parametr

**MLP:** Kazde polaczenie to 1 waga + bias na neuron:
- Wagi: $n_l \times n_{l+1}$
- Biasy: $n_{l+1}$

Tabela pokazuje, ze KAN z mala architektura moze miec mniej parametrow niz MLP.

---

## Komorka [9] - Kod: Wykres skalowania parametrow

**Co robi:** Rysuje wykres log-log pokazujacy jak rosnie liczba parametrow
KAN przy zwiekszaniu rozdzielczosci siatki (G).

**Wniosek:** Parametry KAN rosna liniowo z G (nie eksponencjalnie),
co pozwala na stopniowe zwiekszanie dokladnosci (grid refinement).

---

## Komorka [10] - Markdown: Sekcja 4 naglowek

Naglowek *Hyperparameter Tuning KAN* -- grid search.

---

## Komorka [11] - Kod: Grid search KAN

**Co robi:** Przeszukuje 100 kombinacji hiperparametrow KAN:
- `widths`: rozne architektury ([1,3,1], [1,5,1], [1,5,3,1], [1,8,1])
- `grids`: rozdzielczosci siatki (3, 5, 8, 10, 15)
- `lambs`: sila regularyzacji (0, 0.001, 0.01, 0.1, 1.0)

Dla kazdej kombinacji trenuje KAN i zapisuje R2, MSE, czas.
Sortuje wyniki i wypisuje Top 10.

**Kluczowe koncepty:**
- **lamb (regularyzacja L1):** Promuje rzadkosc -- mniejsze wartosci
  na krawedziach sa zerowane, co upraszcza siec
- **Grid search:** Systematyczne przeszukiwanie przestrzeni hiperparametrow

---

## Komorka [12] - Kod: Wplyw stopnia splajnu k

**Co robi:** Testuje wplyw stopnia splajnu $k \in \{2, 3, 4, 5\}$
na dokladnosc modelu.

**Kluczowe:** Wyzszy k = gladsza funkcja, ale wiecej parametrow.
Zazwyczaj k=3 (kubiczny) jest optymalny -- kompromis miedzy
elastycznoscia a zlozonoscia.

---

## Komorka [13] - Kod: Heatmapa Grid vs Lambda

**Co robi:** Tworzy heatmape R2 w funkcji (G, lambda) dla ustalonej architektury.

**Interpretacja:**
- Os X: rozdzielczosci siatki (G)
- Os Y: sila regularyzacji (lambda)
- Kolor: R2 (im cieplej, tym lepiej)
- Pozwala zidentyfikowac optymalna kombinacje G i lambda

---

## Komorka [14] - Markdown: Sekcja 5 naglowek

Naglowek *Hyperparameter Tuning MLP*.

---

## Komorka [15] - Kod: Grid search MLP + porownanie

**Co robi:** Przeszukuje 30 kombinacji hiperparametrow MLP:
- `hidden_layers`: 5 architektur (od [16] do [128,64,32])
- `activations`: relu, tanh
- `learning_rates`: 0.001, 0.01, 0.1

Nastepnie porownuje najlepszy KAN z najlepszym MLP.

---

## Komorka [16] - Kod: Porownanie najlepszych modeli

**Co robi:** Drukuje zestawienie najlepszego KAN vs najlepszego MLP
po tuningu. Porownuje R2, MSE, liczbe parametrow, czas treningu.

---

## Komorka [17] - Markdown: Sekcja 6 naglowek

Naglowek *Eksperymenty 1D* -- trzy funkcje testowe.

---

## Komorka [18] - Kod: Eksperymenty 1D

**Co robi:** Testuje 4 modele na 3 funkcjach 1D:
- **Funkcje:** Wielomian, Sinus, Zlozona sinusoidalna
- **Modele:** KAN [1,3,1], KAN [1,5,1], MLP [32,16], MLP [64,32,16]

Dla kazdej kombinacji trenuje model, mierzy R2/MSE/czas,
zapisuje predykcje do pozniejszej wizualizacji.

**Cel:** Systematyczne porownanie na prostych, kontrolowanych problemach.

---

## Komorka [19] - Kod: Tabela wynikow 1D

**Co robi:** Formatuje wyniki z komorki 18 jako tabele pandas
z kolumnami: dataset, model, params, R2, MSE, RMSE, MAE, czas.

---

## Komorka [20] - Kod: Wykresy predykcji 1D

**Co robi:** Rysuje 3 wykresy (jeden na funkcje) z:
- Prawdziwa funkcja (linia ciagla)
- Predykcje kazdego modelu (rozne kolory/style)
- Legenda z R2 kazdego modelu

Pozwala wizualnie ocenic jakosc dopasowania.

---

## Komorka [21] - Markdown: Sekcja 7 naglowek

Naglowek *Eksperymenty 2D* -- $f(x_1, x_2) = e^{\sin(\pi x_1) + x_2^2}$.

---

## Komorka [22] - Kod: Eksperymenty 2D

**Co robi:** Testuje modele na trudniejszej funkcji 2-wymiarowej.
Wyzwanie: interakcja miedzy zmiennymi (sin + kwadrat w wykladniku).

Porownuje KAN z roznymi architekturami i MLP.

---

## Komorka [23] - Markdown: Sekcja 8 naglowek

Naglowek *Diabetes* -- prawdziwy zbior danych.

---

## Komorka [24] - Kod: Eksperymenty Diabetes

**Co robi:** Testuje modele na prawdziwym zbiorze Diabetes (sklearn):
- 442 probki, 10 cech (wiek, plec, BMI, cisnienie, itp.)
- Cel: wskaznik progresji cukrzycy po roku

**Znaczenie:** Prawdziwe dane maja szum, korelacje miedzy cechami,
nieliniowosci -- to realistyczny test.

---

## Komorka [25] - Markdown: Sekcja 9 naglowek

Naglowek *Grid refinement KAN* z wzorem $MSE \sim O(G^{-(k+1)})$.

---

## Komorka [26] - Kod: Grid refinement

**Co robi:** Demonstruje unikalna ceche KAN -- stopniowe zwiekszanie
rozdzielczosci siatki (grid refinement).

**Algorytm:**
1. Trenuj KAN z mala siatka (G=3)
2. Uzyj `model.refine(new_grid)` aby zwiekszyc G
3. Dotrenuj na nowej siatce (parametry interpolowane z poprzedniej)
4. Powtorz dla rosnacych G

**Prawo potegowe:** $MSE \sim G^{-(k+1)}$ -- MSE maleje potegowo z G.
Dopasowuje prosta na wykresie log-log (regresja liniowa na log(MSE) vs log(G)).

**Dlaczego to wazne:** MLP nie ma odpowiednika! W MLP trzeba zmienic
architekture i trenowac od nowa. KAN moze plynnie zwiekszac dokladnosc.

---

## Komorka [27] - Markdown: Sekcja 10 naglowek

Naglowek *Analiza czasu treningu i efektywnosci*.

---

## Komorka [28] - Kod: Wykresy czasu i efektywnosci

**Co robi:** Tworzy 3 wykresy podsumowujace:
1. **R2 vs czas treningu** -- scatter plot, KAN vs MLP roznym kolorem
2. **R2 vs liczba parametrow** -- efektywnosc parametryczna
3. **Box plot R2** -- rozklad R2 dla KAN i MLP

**Cel:** Wizualne porownanie kompromisu dokladnosc/czas/zlozonosc.

---

## Komorka [29] - Kod: Podsumowanie statystyk

**Co robi:** Oblicza i drukuje statystyki zbiorcze:
- Srednie/odchylenia R2, MSE, czasu dla KAN i MLP
- Min/max R2 dla obu typow
- Pozwala ocenic ogolna tendencje (np. KAN srednio lepszy ale wolniejszy)

---

## Komorka [30] - Markdown: Sekcja 11 naglowek

Naglowek *Interpretowalnosc KAN*.

---

## Komorka [31] - Kod: Wizualizacja KAN

**Co robi:** Demonstruje interpretowalnosc KAN:

1. **`model.plot()`** -- rysuje graf sieci z nauczonymi funkcjami aktywacji
   na kazdej krawedzi. Kazda krawedz pokazuje ksztalt nauczonego splajnu.

2. **Trzy metryki wizualizacji:**
   - `metric='backward'` -- wagi wsteczne (backward attribution)
   - `metric='forward_n'` -- norma splajnu (rozmiar aktywacji)
   - `metric='forward_u'` -- unikalna wariancja (ile informacji dodaje)

**Kluczowe:** To unikalna cecha KAN -- mozna "zobaczyc" co siec nauczyla sie
na kazdej krawedzi, jakby to byla znana funkcja matematyczna.

---

## Komorka [32] - Markdown: Sekcja 12 naglowek

Naglowek *Pruning sieci KAN*.

---

## Komorka [33] - Kod: Pruning KAN

**Co robi:** Demonstruje pruning (przycinanie) sieci KAN:

1. Trenuje wieksza siec (np. [1, 8, 8, 1])
2. Wywoluje `model.prune()` -- automatycznie usuwa zbedne krawedzie/neurony
3. Porownuje: R2, liczbe parametrow, wizualizacje przed i po

**Kluczowe koncepty:**
- **Pruning:** Usuwanie zbednych polaczen -- siec staje sie mniejsza
  ale zachowuje dokladnosc (lub traci minimalnie)
- PyKAN oferuje tez `prune_node()` i `prune_edge()` do recznego przycinania
- Pruned siec jest bardziej interpretatywna -- mniej krawedzi do analizy

**Dlaczego wazne:** MLP tez mozna prunowac, ale w KAN pruning jest
bardziej naturalny i zachowuje interpretowalnosc.

---

## Komorka [34] - Markdown: Sekcja 13 naglowek

Naglowek *Symbolic fitting*.

---

## Komorka [35] - Kod: Symbolic fitting

**Co robi:** Demonstruje odkrywanie symbolicych formul przez KAN:

1. Trenuje KAN na $f(x,y) = \sin(\pi x) + y^2$
2. `suggest_symbolic(l, i, j)` -- sugeruje symboliczna funkcje
   dla krawedzi (l=warstwa, i=wejscie, j=wyjscie)
3. `auto_symbolic()` -- automatycznie dopasowuje symbole do wszystkich krawedzi
4. `symbolic_formula()` -- drukuje cala odkryta formule

**Kluczowe koncepty:**
- **Symbolic regression:** Odkrywanie formuly matematycznej z danych
- KAN robi to naturalnie bo kazda krawedz to osobna 1D funkcja
- Porownujemy odkryta formule z prawdziwa
- Mozna uzyskac dokladna formule (nie aproksymacje!)

**Przyklad:** Po treningu na $\sin(\pi x) + y^2$ KAN powinien odkryc:
krawedz 1 = sin, krawedz 2 = kwadrat.

---

## Komorka [36] - Markdown: Sekcja 14 naglowek

Naglowek *Feature attribution*.

---

## Komorka [37] - Kod: Feature attribution

**Co robi:** Ocenia waznosc cech wejsciowych za pomoca KAN:

1. Trenuje KAN na zbiorze Diabetes (10 cech)
2. `model.feature_score` -- oblicza wynik waznosci kazdej cechy
3. `model.prune_input()` -- automatycznie usuwa nieistotne cechy
4. Wyswietla ranking cech + wykres slupkowy

**Kluczowe koncepty:**
- **Feature attribution:** Ktore cechy sa najwazniejsze dla predykcji?
- KAN mierzy to przez aktywnosci krawedzi wychodzacych z kazdego wejscia
- `prune_input()` -- agresywnie usuwa cechy o niskim score
- Porownanie z korelacja Pearsona (klasyczna metoda)

**Dlaczego wazne:** Pozwala zrozumiec ktore cechy sa kluczowe,
przydatne w medycynie, nauce, inzynierii.

---

## Komorka [38] - Markdown: Sekcja 15 naglowek

Naglowek *Metryki regularyzacji (reg_metric)*.

---

## Komorka [39] - Kod: Porownanie reg_metric

**Co robi:** Porownuje 4 metryki regularyzacji L1 dostepne w KAN:

1. `edge_forward_spline_n` -- norma splajnu (forward, normalizowana)
2. `edge_forward_spline_u` -- unikalna wariancja splajnu (forward)
3. `edge_forward_sum` -- suma aktywacji (forward)
4. `edge_backward` -- atrybucja wsteczna (backward)

Dla kazdej metryki trenuje KAN z ta sama architektura i porownuje R2.

**Kluczowe koncepty:**
- **reg_metric** wplywa na to *ktore* krawedzie sa regularyzowane
- Rozne metryki moga dac rozne wyniki pruning/interpretowalnosci
- `edge_backward` -- podobne do gradient-based attribution w MLP
- `edge_forward_spline_n` -- domyslna, bazuje na normie splajnu

---

## Komorka [40] - Markdown: Sekcja 16 naglowek

Naglowek *Odpornosc na szum i efektywnosc probkowa*.

---

## Komorka [41] - Kod: Odpornosc na szum

**Co robi:** Testuje jak KAN i MLP radza sobie z rosnacym szumem
($\sigma \in \{0, 0.05, 0.1, 0.2, 0.3, 0.5\}$).

Rysuje wykres R2 vs poziom szumu dla KAN i MLP.

**Hipoteza:** Regularyzacja w KAN (splajny maja inherentna gladkosc)
moze dac lepsza odpornosc na szum niz MLP z ReLU.

---

## Komorka [42] - Kod: Efektywnosc probkowa

**Co robi:** Testuje jak modele radza sobie z mala iloscia danych
($n \in \{50, 100, 200, 300, 500, 800, 1000\}$).

Rysuje wykres R2 vs liczba probek treningowych.

**Hipoteza:** KAN moze byc bardziej efektywny probkowo (sample efficient)
dzieki regularyzacji splajnow i mniejszej liczbie parametrow.

---

## Komorka [43] - Markdown: Sekcja 17

Markdown z 10 glownymi wnioskami pracy:
1. Parametry, 2. HP tuning, 3. 1D, 4. 2D, 5. Diabetes,
6. Grid refinement, 7. Czas, 8. Interpretowalnosc,
9. Szum/probki, 10. Podsumowanie.

---

## Komorka [44] - Kod: Finalna tabela porownawcza

**Co robi:** Tworzy tabele 14 kryteriow porownawczych:
Teoria, Parametry, R2 (1D), R2 (2D), R2 (Diabetes), Czas treningu,
Interpretowalnosc, Grid refinement, Odpornosc na szum,
Efektywnosc probkowa, Pruning, Symbolic fitting,
Feature attribution, HP sensitivity.

Kazde kryterium ma ocene dla KAN i MLP (np. "Tak/Nie",
"Dobra/Ograniczona", opisy jakosciowe).

Na koncu drukuje 3 glowne wnioski.

---

# Slowniczek kluczowych pojec

| Pojecie | Opis |
|---------|------|
| **KAN** | Kolmogorov-Arnold Network -- siec z uczonymi aktywacjami na krawedziach |
| **MLP** | Multi-Layer Perceptron -- klasyczna siec z aktywacjami na neuronach |
| **B-spline** | Funkcja bazowa splajnow, definiowana przez wezly siatki i stopien k |
| **Grid (G)** | Liczba interwalow siatki B-spline -- kontroluje dokladnosc KAN |
| **k (degree)** | Stopien splajnu: k=3 kubiczny, k=5 piatego stopnia |
| **LBFGS** | Optymalizator quasi-Newtonowski, domyslny dla KAN |
| **Grid refinement** | Zwiekszanie G bez retreningu od zera -- unikalna cecha KAN |
| **Pruning** | Usuwanie zbednych krawedzi/neuronow z sieci |
| **Symbolic fitting** | Odkrywanie formuly matematycznej z nauczonego modelu |
| **Feature attribution** | Ranking waznosci cech wejsciowych |
| **reg_metric** | Metryka regularyzacji L1 w KAN (4 warianty) |
| **R2** | Wspolczynnik determinacji: 1 = idealny, 0 = model staly |
| **MSE** | Blad sredniokwadratowy |
| **lamb** | Sila regularyzacji L1 w KAN |
| **DEVICE** | Urzadzenie obliczeniowe: 'cuda' (GPU NVIDIA), 'mps' (Apple), 'cpu' |
