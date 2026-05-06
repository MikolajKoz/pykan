# Teoria KAN vs MLP -- kompendium wiedzy

Ponizszy dokument zawiera pelne wyjasnienie teoretyczne sieci KAN i MLP,
oparte na analizie tutoriali PyKAN, artykulow naukowych i eksperymentow.

---

## 1. Twierdzenie Kolmogorova-Arnolda (1957)

Kazda ciagla funkcja wielowymiarowa $f: [0,1]^n \to \mathbb{R}$ da sie zapisac jako:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

gdzie $\phi_{q,p}$ i $\Phi_q$ sa funkcjami jednej zmiennej.

**Co to znaczy:** Kazda wielowymiarowa funkcja jest zlozeniem jednowymiarowych funkcji.
Nie potrzebujesz funkcji wielu zmiennych -- wystarczy je skomponowac z 1D funkcji.

**Dlaczego to wazne dla sieci neuronowych:** Zamiast uczyc sie wag (jak MLP),
mozna uczyc sie samych funkcji 1D na krawedziach grafu. To wlasnie robi KAN.

---

## 2. Architektura KAN vs MLP -- fundamentalna roznica

### MLP (Multi-Layer Perceptron):
```
neuron = aktywacja( suma(waga_i * wejscie_i) + bias )
```
- **Parametry na polaczeniach:** wagi (liczby skalarne)
- **Aktywacja na neuronach:** stala funkcja (ReLU, tanh, sigmoid)
- Kazdy neuron oblicza: $\sigma(w^T x + b)$

### KAN (Kolmogorov-Arnold Network):
```
wyjscie_krawedzi = nauczona_funkcja_1D(wejscie)
neuron = suma(wyjsc krawedzi)
```
- **Parametry na krawedziach:** cale funkcje (B-spline)
- **Na neuronach:** tylko sumowanie (brak stale ustalonej aktywacji)
- Kazda krawedz oblicza: $\phi(x)$ gdzie $\phi$ to B-spline

### Kluczowa intuicja:
- MLP uczy sie **gdzie** (wagi decyduja ktore wejscia sa wazne)
- KAN uczy sie **jak** (funkcje na krawedziach ucza sie transformacji)

---

## 3. B-spline -- fundament KAN

### Co to B-spline?
Funkcja sklejana (spline) zdefiniowana przez:
- **Siatke (grid):** punkty $t_0, t_1, ..., t_G$ dzielace dziedzine
- **Stopien k:** gladkosc splajnu (k=3 = kubiczny, ciaglosc C2)
- **Bazy B-spline:** $B_{i,k}(x)$ -- lokalne funkcje bazowe

Kazda krawedz KAN oblicza:
$$\phi(x) = \sum_{i=0}^{G+k-1} c_i \cdot B_{i,k}(x)$$

### Dlaczego B-spline a nie wielomiany?
- **Lokalna kontrola:** Zmiana jednego wspolczynnika $c_i$ wplywa
  tylko na maly fragment funkcji (nie cala)
- **Stabilnosc numeryczna:** Brak efektu Rungego (oscylacje wielomianow)
- **Gladkosc:** Gwarantowana ciaglosc pochodnych do rzedu k-1
- **Efektywnosc:** Obliczanie w O(k) dzieki rekurencji de Boora

### Ile parametrow na krawedz?
$G + k + 1$ wspolczynnikow, gdzie:
- G = liczba interwalow siatki
- k = stopien splajnu
- +1 = dodatkowy wspolczynik bazowy (residual: $c_0 \cdot b(x)$ w PyKAN tu `base_fun`)

---

## 4. Grid refinement -- unikalna przewaga KAN

### Na czym polega?
1. Trenuj KAN z mala siatka (np. G=3) -- szybki, zgrubny model
2. Zwieksz G (np. na 5) uzywajac `model.refine(5)`:
   - Nowe wspolczynniki sa **interpolowane** ze starych
   - Model nie traci tego co juz nauczyl sie
3. Dotrenuj na nowej, gestszej siatce
4. Powtarzaj az do zadowalajacego bledu

### Prawo potegowe (neural scaling law):
$$MSE \sim O(G^{-(k+1)})$$

Blad maleje potegowo ze wzrostem G. Na wykresie log-log to linia prosta
o nachyleniu $-(k+1)$.

### Dlaczego MLP tego nie ma?
W MLP "rozdzielczosc" to liczba neuronow -- nie mozna plynnie zwiekszyc
dokladnosci bez przebudowy architectury i treningu od zera.
KAN pozwala na **plynne skalowanie** dokladnosci.

### Z tutoriali (Example_1):
```python
grids = [5, 10, 20, 50, 100]
for g in grids:
    model.refine(g)
    model.fit(dataset, steps=50)
```
Kazdy `refine()` zachowuje wiedze i dodaje nowa rozdzielczosc.

---

## 5. Regularyzacja L1 i rzadkosc w KAN

### Problem:
Siec KAN z duza architektura ma wiele krawedzi -- wiekszsc moze byc zbedna.

### Rozwiazanie -- regularyzacja L1:
Dodajemy kare za "wielkosc" aktywacji na krawedziach:
$$Loss = MSE + \lambda \sum_{krawedzie} ||\phi_{ij}||$$

Parametr `lamb` w PyKAN kontroluje sile tej kary.

### 4 metryki regularyzacji (reg_metric) z API_8:

1. **`edge_forward_spline_n`** (domyslna):
   - Norma wspolczynnikow splajnu znormalizowana wejsciem
   - Mierzy "wielkosc" nauczonej funkcji niezaleznie od skali danych

2. **`edge_forward_spline_u`**:
   - Unikalna wariancja -- ile _nowej_ informacji wnosi krawedz
   - Kara te krawedzie ktore sa redundantne

3. **`edge_forward_sum`**:
   - Suma wartosci bezwzglednych aktywacj na danych treningowych
   - Najprostsza -- patrzy na faktyczne wyjscia

4. **`edge_backward`**:
   - Atrybucja wsteczna (jak gradient * aktywacja)
   - Podobne do Grad-CAM w CNN -- mierzy _wplyw_ na wyjscie

### Z tutoriali (API_8):
Kazda metryka daje inne priorytety pruning -- `edge_backward` lepiej
identyfikuje przyczynowe polaczenia, `spline_n` lepiej regularyzuje rozmiar.

---

## 6. Pruning -- przycinanie sieci

### Filozofia:
"Trenuj duza siec, potem przytnij do esencji."

### Jak dziala `model.prune()` (API_7):
1. Oblicza score kazdej krawedzi (wg wybranej metryki)
2. Krawedzie ponizej progu sa usuwane
3. Neurony bez wejsc lub wyjsc sa usuwane
4. Wynik: mniejsza, interpretatywna siec

### Rodzaje pruning:
- `model.prune()` -- automatyczny (prog na podstawie statystyk)
- `model.prune_node(l, i)` -- usun konkretny neuron
- `model.prune_edge(l, i, j)` -- usun konkretna krawedz

### Dlaczego to wazne?
Po pruning siec ma 3-5 krawedzi zamiast 20+, co pozwala:
- Wizualnie interpretowac strukture
- Zastosowac symbolic fitting (mniej krawedzi do analizy)
- Zmniejszyc overfitting

---

## 7. Symbolic fitting -- odkrywanie formul

### Cel:
Zamienic nauczona siec KAN w **dokladna formule matematyczna**.

### Jak to dziala (Example_5, Example_9):
1. Trenuj KAN na danych (np. $f(x,y) = \sin(\pi x) + y^2$)
2. `suggest_symbolic(l, i, j)` -- dopasowuje biblioteke funkcji
   (sin, cos, exp, log, x^2, ...) do nauczonego splajnu na krawedzi (l, i, j)
3. `fix_symbolic(l, i, j, 'sin')` -- zamienia splajn na dokladna funkcje
4. `symbolic_formula()` -- drukuje cala odkryta formule

### Biblioteka funkcji:
PyKAN domyslnie proba: x, x^2, x^3, x^4, sqrt, exp, log, sin, cos, tan, tanh,
abs, sign, arcsin, arccos, arctan, Bessel, itp.

### Dlaczego to przelomowe?
- **MLP jest czarna skrzynka** -- nie da sie odczytac formuly
- **KAN jest biala skrzynka** -- kazda krawedz to osobna 1D funkcja,
  ktora mozna zidentyfikowac symbolizne
- To jak automatyczny "naukowiec" -- odkrywa prawa fizyki z danych

### Z tutoriali (Example_9 -- osobliwosci):
Dla funkcji z osobliwosciami (np. 1/x) uzywamy `singularity_avoiding=True`
co dodaje specjalne bazy zdolne modelowac bieguny.

---

## 8. Feature attribution -- waznosc cech

### Problem:
Masz 10+ cech wejsciowych. Ktore sa naprawde wazne?

### Jak KAN to rozwiazuje (Interp_4):
1. `model.feature_score` -- oblicza score importancji kazdego wejscia
   na podstawie sumy aktywnosci krawedzi wychodzacych z danego neuronu wejsciowego
2. `model.prune_input()` -- automatycznie usuwa cechy o niskim score

### Czym to sie rozni od MLP feature importance?
- **MLP:** Permutation importance (wolne, aproksymacyjne), SHAP (dokladne ale O(2^n))
- **KAN:** Wbudowane w architekture -- score jest natychmiastowy,
  bo krawedzie _sa_ transformacjami cech

### Zastosowanie:
- Selekcja cech przed modelowaniem
- Interpretacja medyczna (ktore biomarkery wazne?)
- Odkrywanie fizyki (ktore zmienne wchodza do rownania?)

---

## 9. Optymalizacja LBFGS vs Adam

### Dlaczego KAN uzywa LBFGS?
- **LBFGS** (Limited-memory BFGS) to metoda quasi-Newtonowska
- Uzywa aproksymacji macierzy Hessian (drugich pochodnych)
- Zbieznosc kwadratowa (vs liniowa w Adam/SGD)
- Wymaga petnego przebiegu po danych (full-batch)

### Dlaczego MLP uzywa Adam?
- **Adam** to metoda pierwszego rzedu z momentem adaptacyjnym
- Dziala dobrze z mini-batchami i duzymi zbiorami danych
- Skaluje sie do milionow parametrow
- LBFGS bylby zbyt kosztowny pamieciowo dla duzych MLP

### Praktyczny kompromis:
KAN zazwyczaj ma 100-10000 parametrow -- LBFGS jest idealny.
MLP ma 1000-1000000 parametrow -- Adam jest konieczny.

---

## 10. Liczba parametrow -- szczegolowe porownanie

### KAN: $P_{KAN} = \sum_{l=0}^{L-1} n_l \cdot n_{l+1} \cdot (G + k + 1)$
- Architektura [2, 5, 1], G=5, k=3: $2 \times 5 \times 9 + 5 \times 1 \times 9 = 90 + 45 = 135$

### MLP: $P_{MLP} = \sum_{l=0}^{L-1} (n_l \cdot n_{l+1} + n_{l+1})$
- Architektura [2, 32, 16, 1]: $2 \times 32 + 32 + 32 \times 16 + 16 + 16 \times 1 + 1 = 64+32+512+16+16+1 = 641$

### Kluczowy insight:
KAN z mala architektura (np. [2,5,1]) moze osiagnac podobna dokladnosc
co MLP z duzo wieksza ([2,32,16,1]). To dlatego ze kazda krawedz KAN
jest "bogatsza" -- to cala funkcja, nie jeden skalar.

### Skalowanie:
- KAN: parametry rosna liniowo z G (mozna kontrolowac)
- MLP: parametry rosna kwadratowo z szerokoscia warstw

---

## 11. Interpretowalnosc -- dlaczego KAN jest "szklanym pudlem"

### 3 poziomy interpretowalnosci KAN:

**Poziom 1: Wizualizacja grafu**
- `model.plot()` rysuje siec z ksztaltami splajnow na krawedziach
- Od razu widac: ktore krawedzie sa "aktywne" (nietrywialne ksztalty)
- Ktore sa "puste" (bliskie 0 lub liniowe)

**Poziom 2: Identyfikacja funkcji**
- `suggest_symbolic()` dopasowuje znane funkcje
- Mozna przeczytac: "ta krawedz to sinus, ta to kwadrat"
- Powstaje czytelna formalna matematyczna

**Poziom 3: Odkrywanie naukowe**
- Kompletna symbolic formula odkryta z danych
- Mozliwe do weryfikacji analitycznej
- Przyklad: odkrycie praw fizyki z danych eksperymentalnych

### MLP nie ma zadnego z tych poziomow:
- Wagi sa nieczytelne (tysiace losowych liczb)
- Post-hoc metody (LIME, SHAP) sa przyblizone
- Nie mozna odzyskac formuly analitycznej

---

## 12. Odpornosc na szum

### Dlaczego KAN moze byc bardziej odporny?
1. **Gladkosc B-spline:** Splajn stopnia k jest z natury gladki (C^{k-1}),
   co dziala jak wbudowany filtr niskich czestotliwosci
2. **Regularyzacja L1:** Promuje rzadkosc -- uswa szumowe krawedzie
3. **Mniej parametrow:** Mniej zdolnosc do zapamieta szumu (mniej overfitting)

### Dlaczego MLP moze byc bardziej wrazliwy?
1. **ReLU nie jest gladkie:** Fragmenty liniowe moga dopasowacc sie do szumu
2. **Duza pojemnosc:** Wiecej parametrow = wieksze ryzyko overfitting
3. **Regularyzacja zewnetrzna:** Dropout, weight decay -- dodawane recznie

---

## 13. Efektywnosc probkowa (sample efficiency)

### Teza:
KAN potrzebuje mniej danych do dobrego dopasowania niz MLP.

### Dlaczego?
1. **Mateljszy model:** Mniej parametrow = mniej danych potrzeba
2. **Indukcyjne prior:** B-spline zaklada gladkosc -- potrzeba mniej
   punktow do okreslenia gladkiej funkcji niz dowolnej
3. **Grid refinement:** Mozna zaczac od G=3 (malo param) i stopniowo zwiekszac

### Kiedy MLP wygrywa?
- Bardzo duze zbiory danych (>10000 probek)
- Dane o wysokiej wymiarowosci (>100 cech)
- Gdy potrzeba szybkiego treningu (mini-batch SGD skaluje sie lepiej)

---

## 14. Wady KAN -- uczciwa ocena

1. **Czas treningu:** LBFGS + full-batch = wolniejszy niz Adam + mini-batch
2. **Skalowanie:** Nie sprawdzony na duzych problemach (ImageNet, NLP)
3. **Ekosystem:** Mloda biblioteka, mniej narzedzi/wsparcia niz PyTorch/TF
4. **Wymiarowosc:** Przeklenstwo wymiarowosci -- wiecej cech = wiecej krawedzi
5. **Stabilnosc:** LBFGS moze byc niestabilny (line search failures)
6. **Brak mini-batch:** Caly zbior w pamieci (problemy z duzymi danymi)

---

## 15. Kiedy uzywac KAN vs MLP?

### Uzyj KAN gdy:
- Dane sa niskowowymiarowe (1-20 cech)
- Zalezy ci na interpretowalnosci
- Chcesz odkryc formule matematyczna
- Masz malo danych (<1000 probek)
- Problem jest "naukowy" (fizyka, chemia, biologia)

### Uzyj MLP gdy:
- Dane sa wysokowymiarowe (obrazy, tekst, >100 cech)
- Potrzebujesz szybkiego treningu
- Masz duzo danych (>10000 probek)
- Interpretowalnosc nie jest priorytetem
- Potrzebujesz sprawdzonego, stabilnego rozwiazania

---

## 16. Podsumowanie najwazniejszych lekcji z tutoriali PyKAN

| Tutorial | Kluczowa lekcja |
|----------|----------------|
| Example_1 | Grid refinement daje neural scaling laws -- MSE ~ G^{-(k+1)} |
| Example_3 | Glebsze KAN (3 warstwy) lepiej modeluja zlozone kompozycje |
| Example_5 | suggest_symbolic() potrafi zidentyfikowac Bessela, sin, exp |
| Example_9 | singularity_avoiding=True dla funkcji z biegunami |
| Example_11 | base_fun='identity' + lamb_coef do wymuszania liniowosci |
| API_2 | model.plot(beta=) kontroluje gladkosc wizualizacji |
| API_4 | sparse_init=True daje lepszy punkt startowy |
| API_5 | G+k wspolczynnikow, grid_eps dla stabilnosci brzegowej |
| API_6 | lamb, lamb_entropy, seed -- klucz do powtarzalnosci |
| API_7 | prune() -> prune_node() -> prune_edge() -- hierarchia pruning |
| API_8 | 4 metryki reg_metric daja rozne wyniki regularyzacji |
| API_10 | .to(device) przenosi model na GPU (identycznie jak PyTorch) |
| API_11 | create_dataset(f, ranges) -- standardowy format danych PyKAN |
| Interp_1 | MultKAN: width=[2,[5,2],1] -- mnozenie w warstwie |
| Interp_3 | kanpiler -- kompilacja formuly do KAN (odwrotnosc symbolic fitting) |
| Interp_4 | feature_score + prune_input() -- selekcja cech |
| Interp_11 | sparse_init pomaga w interpretowalnosci od startu |

---

## 17. Dlaczego robimy te eksperymenty?

### Cel naukowy pracy:
Odpowiedziec na pytanie: **Czy KAN jest lepszy od MLP w regresji?**

### Metodologia:
Nie mozna odpowiedziec jednym eksperymentem. Trzeba zbadac:

1. **Rozne wymiarowosci** (1D, 2D, 10D) -- czy przewaga KAN zalezy od wymiaru?
2. **Rozne funkcje** (latwe, trudne, realne) -- czy KAN lepiej modeluje pewne typy?
3. **Rozne metryki** (R2, MSE, czas, params) -- co jest "lepsze"?
4. **Rozne warunki** (szum, malo danych) -- kiedy kazdy model sie "lamie"?
5. **Unikalne cechy KAN** (grid refinement, pruning, symbolic) -- co KAN dodaje?

### Wnioski musza byc:
- **Uczciwe:** Oba modele z tunigiem (nie porownywac tuned KAN z default MLP)
- **Powtarzalne:** Seed, jasne parametry, identyczne dane
- **Kompletne:** Nie tylko "kto wygrywa" ale "kiedy i dlaczego"
