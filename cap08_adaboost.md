---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 8. Boosting: demostración del Algoritmo AdaBoost

## Enunciado

```{figure} images/enunciado_adaboost_a.png
:width: 95%

Planteamiento del clasificador AdaBoost, Ecuaciones (5.2)–(5.3).
```

```{figure} images/enunciado_adaboost_b.png
:width: 80%

Algoritmo 5.1 (AdaBoost). Se pide **demostrar** el algoritmo.
```

**Planteamiento.** Hay $N$ observaciones $(y_n,\boldsymbol{x}_n)$ con $y_n\in\{-1,1\}$, y se busca

$$
f(\boldsymbol{x})=\operatorname{sgn}\{F(\boldsymbol{x})\},\qquad F(\boldsymbol{x}):=\sum_{k=1}^{K}a_k\phi(\boldsymbol{x};\boldsymbol{\theta}_k),\qquad \phi(\boldsymbol{x};\boldsymbol{\theta}_k)\in\{-1,1\}.
$$

El ajuste es **greedy por etapas**: con $F_{i}(\cdot)=F_{i-1}(\cdot)+a_i\phi(\cdot;\boldsymbol{\theta}_i)$ y $F_0\equiv0$, en la iteración $i$ se resuelve, con la **pérdida exponencial** $\mathcal{L}(y,F(\boldsymbol{x}))=\exp(-yF(\boldsymbol{x}))$,

$$
(a_{i}, \boldsymbol{\theta}_{i})=\underset{a,\boldsymbol{\theta}}{\operatorname{argmin}}\ J(a,\boldsymbol{\theta}),\qquad J(a,\boldsymbol{\theta}):=\sum_{n=1}^{N}\exp\left(-y_{n}\big(F_{i-1}(\boldsymbol{x}_{n})+a\phi(\boldsymbol{x}_{n};\boldsymbol{\theta})\big)\right).
$$ (eq-54)

**Por demostrar.** Que cada paso del Algoritmo 5.1 es la solución exacta de {eq}`eq-54`:

- **(A)** la factorización con los pesos $\omega_n^{(i)}$, Ecuación (5.5);
- **(B)** para $a>0$, $\boldsymbol{\theta}_i$ minimiza el error ponderado $P_i$, Ecuación (5.6), y se cumple (5.7);
- **(C)** $a_i=\frac12\ln\frac{1-P_i}{P_i}$ es el minimizador en $a$;
- **(D)** la recursión de pesos con normalización $Z_i$ (pasos 4–8 del algoritmo).

Además se demuestra la afirmación de las notas de que **el error de entrenamiento tiende a cero exponencialmente rápido** (Teorema de Freund y Schapire).

## 8.1 (A) Factorización en pesos

```{prf:proposition} Ecuación (5.5)
:label: prop-55

Con $\omega_{n}^{(i)}:=\exp(-y_{n}F_{i-1}(\boldsymbol{x}_{n}))$,

$$
J(a,\boldsymbol{\theta})=\sum_{n=1}^{N}\omega_n^{(i)}\exp\big(-y_na\phi(\boldsymbol{x}_n;\boldsymbol{\theta})\big),
$$

y $\omega_n^{(i)}>0$ no depende de $(a,\boldsymbol{\theta})$. Por lo tanto, para $a$ fijo, $\boldsymbol{\theta}_i=\operatorname{argmin}_{\boldsymbol{\theta}}\sum_n\omega_n^{(i)}\exp(-y_na\phi(\boldsymbol{x}_n;\boldsymbol{\theta}))$. Multiplicar todos los $\omega_n^{(i)}$ por una misma constante positiva no cambia los argmin.
```

```{prf:proof}
Por la propiedad $e^{u+v}=e^{u}e^{v}$,

$$
\exp\!\big(-y_n(F_{i-1}(\boldsymbol{x}_n)+a\phi(\boldsymbol{x}_n;\boldsymbol{\theta}))\big)=\underbrace{\exp(-y_nF_{i-1}(\boldsymbol{x}_n))}_{\omega_n^{(i)}}\ \exp(-y_na\phi(\boldsymbol{x}_n;\boldsymbol{\theta})).
$$

$F_{i-1}$ está fijo (fue calculado en las iteraciones anteriores), así que $\omega_n^{(i)}$ es una constante positiva. Si $\omega_n^{(i)}$ se reemplaza por $\omega_n^{(i)}/W$ con $W>0$, la función objetivo queda multiplicada por $1/W$, y $\operatorname{argmin}(g/W)=\operatorname{argmin}(g)$.
```

En adelante se usan pesos **normalizados**, $\sum_{n}\omega_n^{(i)}=1$, como en el algoritmo. Con $\omega_n^{(1)}=1/N$ y $F_0\equiv0$, esto es consistente: $\exp(0)=1$, normalizado, da $1/N$.

## 8.2 (B) Reducción al error ponderado $P_i$

Como $y_n\in\{-1,1\}$ y $\phi\in\{-1,1\}$, el producto $y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta})$ vale $+1$ si la observación está bien clasificada y $-1$ si está mal clasificada (nunca $0$). La función $\chi_{(-\infty,0]}(y_n\phi)$ vale entonces $1$ exactamente en los errores.

```{prf:proposition} Ecuaciones (5.6) y (5.7)
:label: prop-56

Con $P(\boldsymbol{\theta}):=\sum_{n=1}^{N}\omega_n^{(i)}\chi_{(-\infty,0]}(y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta}))$ y pesos normalizados,

$$
J(a,\boldsymbol{\theta})=e^{-a}\big(1-P(\boldsymbol{\theta})\big)+e^{a}P(\boldsymbol{\theta})=e^{-a}+\big(e^{a}-e^{-a}\big)P(\boldsymbol{\theta}).
$$ (eq-J-P)

En consecuencia:

1. Si $a>0$, $\ \operatorname{argmin}_{\boldsymbol{\theta}}J(a,\boldsymbol{\theta})=\operatorname{argmin}_{\boldsymbol{\theta}}P(\boldsymbol{\theta})$, y este último **no depende de $a$**. Esa es la Ecuación (5.6), con $P_i:=P(\boldsymbol{\theta}_i)$.
2. En el óptimo, $\displaystyle\sum_{y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i)<0}\omega_n^{(i)}=P_i$ y $\displaystyle\sum_{y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i)>0}\omega_n^{(i)}=1-P_i$, que es la Ecuación (5.7).
```

```{prf:proof}
Partimos el índice $n$ en bien clasificados ($y_n\phi=1$, donde $e^{-ay_n\phi}=e^{-a}$) y mal clasificados ($y_n\phi=-1$, donde $e^{-ay_n\phi}=e^{a}$):

$$
J(a,\boldsymbol{\theta})=e^{-a}\!\!\sum_{y_n\phi>0}\!\!\omega_n^{(i)}+e^{a}\!\!\sum_{y_n\phi<0}\!\!\omega_n^{(i)}.
$$

La segunda suma es $P(\boldsymbol{\theta})$ por definición. Como los pesos suman $1$ y cada $n$ está en exactamente uno de los dos grupos, la primera suma es $1-P(\boldsymbol{\theta})$. Esto prueba {eq}`eq-J-P` y el punto 2.

Para el punto 1: si $a>0$, entonces $e^{a}-e^{-a}=2\sinh a>0$, de modo que $J(a,\cdot)$ es una función **afín creciente** de $P(\cdot)$ más una constante. Minimizar $J$ en $\boldsymbol{\theta}$ equivale a minimizar $P$, y el minimizador no depende de $a$. Esto justifica el orden del algoritmo: primero se calcula $\boldsymbol{\theta}_i$ y después $a_i$.
```

```{admonition} ¿Y si $a<0$?
:class: note
Si $a<0$, el mismo cálculo muestra que habría que **maximizar** $P$. Pero usar $(a,\phi)$ con $a<0$ es lo mismo que usar $(-a,-\phi)$, y el clasificador $-\phi$ tiene error $1-P$. Por eso basta con considerar $a>0$ y clasificadores base con $P_i<1/2$ (mejores que el azar), que es lo que se espera en la práctica, como dicen las notas.
```

## 8.3 (C) El coeficiente óptimo $a_i$

```{prf:proposition} Coeficiente $a_i$
:label: prop-ai

Si $0<P_i<1$, la función $g(a):=e^{-a}(1-P_i)+e^{a}P_i$ es estrictamente convexa en $\mathbb{R}$ y tiene un único minimizador,

$$
a_i=\frac12\ln\frac{1-P_i}{P_i},
$$

con valor mínimo $g(a_i)=2\sqrt{P_i(1-P_i)}$. Además, $a_i>0\iff P_i<\tfrac12$.
```

```{prf:proof}
*Derivadas.* $g'(a)=-e^{-a}(1-P_i)+e^{a}P_i$ y $g''(a)=e^{-a}(1-P_i)+e^{a}P_i>0$, porque ambos términos son positivos. Así, $g$ es estrictamente convexa y un punto crítico es su mínimo global único.

*Punto crítico.* Multiplicando $g'(a)=0$ por $e^{a}>0$:

$$
e^{2a}P_i=1-P_i\iff e^{2a}=\frac{1-P_i}{P_i}\iff a=\frac12\ln\frac{1-P_i}{P_i}.
$$

*Valor mínimo.* En $a_i$, $e^{a_i}=\sqrt{(1-P_i)/P_i}$ y $e^{-a_i}=\sqrt{P_i/(1-P_i)}$, luego

$$
\begin{aligned}
g(a_i)&=(1-P_i)\sqrt{\frac{P_i}{1-P_i}}+P_i\sqrt{\frac{1-P_i}{P_i}}\\
&=\sqrt{P_i(1-P_i)}+\sqrt{P_i(1-P_i)}=2\sqrt{P_i(1-P_i)}.
\end{aligned}
$$

*Signo.* $a_i>0\iff\frac{1-P_i}{P_i}>1\iff P_i<\frac12$.

*Casos borde.* Si $P_i=0$ (clasificador base perfecto), $g(a)=e^{-a}$ decrece sin mínimo ($a_i\to\infty$), y en la práctica se detiene el algoritmo. Si $P_i\ge\frac12$, el clasificador base no es mejor que el azar y también se detiene (o se invierte, según la nota anterior).
```

```{admonition} Observación (notación del Algoritmo 5.1)
:class: warning
En el paso 3 del algoritmo dice $a_i=1/2[\ln(1-P_i)/P_i]$. Lo correcto, según la derivación, es $a_i=\frac12\ln\!\big(\frac{1-P_i}{P_i}\big)$: el cociente va **dentro** del logaritmo. Del mismo modo, en la inicialización debe leerse $\omega_n^{(1)}=1/N$, $n=1,\dots,N$ (el índice es $n$, no $i$).
```

## 8.4 (D) Recursión de pesos y normalización $Z_i$

```{prf:proposition} Actualización de pesos
:label: prop-pesos

Sean los pesos normalizados $\omega_n^{(i)}\propto\exp(-y_nF_{i-1}(\boldsymbol{x}_n))$. Entonces

$$
\begin{gathered}
\omega_{n}^{(i+1)}=\frac{\exp(-y_{n}F_{i}(\boldsymbol{x}_{n}))}{\sum_{k}\exp(-y_{k}F_{i}(\boldsymbol{x}_{k}))}=\\
=\frac{\omega_{n}^{(i)}\exp(-y_{n}a_{i}\phi(\boldsymbol{x}_{n}; \boldsymbol{\theta}_{i}))}{Z_{i}},\\
Z_{i}=\sum_{n=1}^{N}\omega_{n}^{(i)}\exp(-y_{n}a_{i}\phi(\boldsymbol{x}_{n}; \boldsymbol{\theta}_{i})),
\end{gathered}
$$

que es exactamente lo que calculan los pasos 4–8 del Algoritmo 5.1. Además:

1. $Z_i=2\sqrt{P_i(1-P_i)}<1$ si $P_i\ne\frac12$.
2. Los pesos de las observaciones **mal** clasificadas por $\phi(\cdot;\boldsymbol{\theta}_i)$ se multiplican por $\frac{1}{2P_i}>1$, y los de las **bien** clasificadas por $\frac{1}{2(1-P_i)}<1$ (cuando $P_i<\frac12$).
3. Con los nuevos pesos, el clasificador recién añadido tiene error ponderado **exactamente** $\frac12$: $\sum_{y_n\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i)<0}\omega_n^{(i+1)}=\frac12$.
```

```{prf:proof}
*Recursión.* Como $F_i=F_{i-1}+a_i\phi(\cdot;\boldsymbol{\theta}_i)$, se tiene $\exp(-y_nF_i(\boldsymbol{x}_n))=\exp(-y_nF_{i-1}(\boldsymbol{x}_n))\exp(-y_na_i\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i))$. Si $\omega_n^{(i)}=\exp(-y_nF_{i-1}(\boldsymbol{x}_n))/W_{i}$, multiplicar por $\exp(-y_na_i\phi)$ y renormalizar da $\omega_n^{(i+1)}$; la constante $W_i$ se cancela en la normalización. El denominador que hace sumar $1$ es justamente $Z_i$. En el algoritmo: el paso 5.1 calcula los numeradores, el paso 5.2 acumula $Z_i$ y el paso 7.1 divide.

*1.* $Z_i=J(a_i,\boldsymbol{\theta}_i)$ con pesos normalizados, que por {eq}`eq-J-P` y la {prf:ref}`prop-ai` es $g(a_i)=2\sqrt{P_i(1-P_i)}$. Como $4P(1-P)=1-(1-2P)^{2}\le1$, con igualdad solo si $P=\frac12$, se tiene $Z_i\le1$.

*2.* Si la observación está mal clasificada, el factor es $e^{a_i}/Z_i=\sqrt{\tfrac{1-P_i}{P_i}}\big/\big(2\sqrt{P_i(1-P_i)}\big)=\frac{1}{2P_i}$. Si está bien clasificada, el factor es $e^{-a_i}/Z_i=\frac{1}{2(1-P_i)}$.

*3.* Por el punto 2 y (5.7), $\sum_{\text{mal}}\omega_n^{(i+1)}=\frac{1}{2P_i}\sum_{\text{mal}}\omega_n^{(i)}=\frac{P_i}{2P_i}=\frac12$.
```

```{admonition} Interpretación
:class: tip
El punto 3 formaliza la frase de las notas de que "las muestras difíciles ganan importancia": tras cada iteración, los pesos se reajustan de modo que el último clasificador base queda **en el nivel del azar**. Eso obliga a que el siguiente clasificador base aporte información nueva y no repita al anterior.
```

## 8.5 El error de entrenamiento decae exponencialmente

```{prf:theorem} Freund y Schapire (1997)
:label: thm-fs

Sea $f=\operatorname{sgn}(F_K)$ la salida del Algoritmo 5.1 tras $K$ iteraciones, y $\gamma_i:=\frac12-P_i$. Entonces

$$
\begin{aligned}
\frac1N\sum_{n=1}^{N}\mathbb{1}_{[f(\boldsymbol{x}_n)\neq y_n]}\ &\le\ \prod_{i=1}^{K}Z_i\\
&\ =\ \prod_{i=1}^{K}2\sqrt{P_i(1-P_i)}\ =\ \prod_{i=1}^{K}\sqrt{1-4\gamma_i^{2}}\\
&\le\ \exp\Big(-2\sum_{i=1}^{K}\gamma_i^{2}\Big).
\end{aligned}
$$

En particular, si todos los clasificadores base son "débiles" con margen $\gamma_i\ge\gamma>0$, el error de entrenamiento es $\le e^{-2K\gamma^{2}}$, y es **exactamente cero** en cuanto $K>\frac{\ln N}{2\gamma^{2}}$.
```

```{prf:proof}
*Paso 1: la pérdida exponencial acota la 0-1.* Si $f(\boldsymbol{x}_n)\neq y_n$, entonces $y_nF_K(\boldsymbol{x}_n)\le0$ y $\exp(-y_nF_K(\boldsymbol{x}_n))\ge1$. Si no hay error, la indicadora vale $0\le\exp(\cdot)$. En ambos casos, $\mathbb{1}_{[f(\boldsymbol{x}_n)\neq y_n]}\le\exp(-y_nF_K(\boldsymbol{x}_n))$. Si $F_K(\boldsymbol{x}_n)=0$, contamos un error y la cota sigue valiendo, porque $e^{0}=1$.

*Paso 2: desenrollar la recursión.* Aplicando la {prf:ref}`prop-pesos` $K$ veces desde $\omega_n^{(1)}=1/N$:

$$
\omega_n^{(K+1)}=\frac{1}{N}\cdot\frac{\prod_{i=1}^{K}\exp(-y_na_i\phi(\boldsymbol{x}_n;\boldsymbol{\theta}_i))}{\prod_{i=1}^{K}Z_i}=\frac{\exp(-y_nF_K(\boldsymbol{x}_n))}{N\prod_{i=1}^{K}Z_i}.
$$

Como los pesos suman $1$, sumando en $n$:

$$
\frac1N\sum_{n=1}^{N}\exp(-y_nF_K(\boldsymbol{x}_n))=\prod_{i=1}^{K}Z_i .
$$

*Paso 3: combinar.* Promediando el Paso 1 en $n$ y usando el Paso 2 se obtiene la primera desigualdad. La igualdad $Z_i=2\sqrt{P_i(1-P_i)}$ es la {prf:ref}`prop-pesos`, y $4P_i(1-P_i)=4(\tfrac12-\gamma_i)(\tfrac12+\gamma_i)=1-4\gamma_i^{2}$.

*Paso 4: cota exponencial.* Para todo $u$, $1+u\le e^{u}$. Con $u=-4\gamma_i^{2}$, $\sqrt{1-4\gamma_i^{2}}\le\sqrt{e^{-4\gamma_i^{2}}}=e^{-2\gamma_i^{2}}$. Multiplicando sobre $i$ se obtiene la última cota.

*Paso 5: error nulo.* El error de entrenamiento es un múltiplo de $1/N$. Si $e^{-2K\gamma^{2}}<1/N$, es decir, si $K>\ln N/(2\gamma^{2})$, el error es menor que $1/N$ y, por lo tanto, cero.
```

```{admonition} Observación
:class: note
El teorema habla del error de **entrenamiento**. Como advierten las notas, que este error tienda a cero **no** implica que el error de prueba también lo haga; el comportamiento de generalización de AdaBoost se estudia con márgenes y complejidad de la clase base.
```

## 8.6 Verificación numérica

Se implementa el Algoritmo 5.1 tal cual, con *stumps* (árboles de profundidad 1) como clasificadores base, sobre datos sintéticos con frontera **oblicua** $x_1+x_2=0$. Ningún stump, que solo corta en paralelo a los ejes, la resuelve por sí solo, pero su combinación sí. En cada iteración se verifica:

- (C) que $a_i$ minimiza numéricamente $g(a)$;
- (D) que $Z_i=2\sqrt{P_i(1-P_i)}$ y que el error ponderado del último clasificador con los pesos nuevos es $1/2$;
- el {prf:ref}`thm-fs`: error de entrenamiento $\le\prod Z_i\le\exp(-2\sum\gamma_i^2)$.

```{code-cell} ipython3
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(0)
N = 400
X = rng.normal(size=(N, 2))
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)                   # frontera oblicua: no la resuelve un solo stump

w = np.full(N, 1/N)                                          # inicialización: w_n^(1) = 1/N
F = np.zeros(N); K = 300
hist = {"P": [], "Z": [], "err": [], "prodZ": [], "cota": []}
chk_a = chk_Z = chk_half = 0.0
for i in range(K):
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)   # theta_i: minimiza P_i
    phi = stump.predict(X)
    P = w[phi != y].sum()                                                    # error ponderado P_i
    a = 0.5*np.log((1 - P)/P)                                                # paso 3
    g = lambda t: np.exp(-t)*(1 - P) + np.exp(t)*P
    chk_a = max(chk_a, abs(minimize_scalar(g).x - a))                        # (C): a_i es el argmin
    w_new = w*np.exp(-y*a*phi); Z = w_new.sum(); w_new /= Z                  # pasos 4-8
    chk_Z = max(chk_Z, abs(Z - 2*np.sqrt(P*(1 - P))))                        # (D.1)
    chk_half = max(chk_half, abs(w_new[phi != y].sum() - 0.5))               # (D.3)
    w = w_new; F += a*phi
    hist["P"].append(P); hist["Z"].append(Z)
    hist["err"].append(np.mean(np.sign(F) != y))
    hist["prodZ"].append(np.prod(hist["Z"]))
    hist["cota"].append(np.exp(-2*np.sum((0.5 - np.array(hist["P"]))**2)))

err, prodZ, cota = map(np.array, (hist["err"], hist["prodZ"], hist["cota"]))
print(f"máx |argmin numérico de g - a_i|          = {chk_a:.2e}")
print(f"máx |Z_i - 2 sqrt(P_i(1-P_i))|            = {chk_Z:.2e}")
print(f"máx |error ponderado nuevo de phi_i - 1/2| = {chk_half:.2e}")
print("¿error <= prod Z_i en todas las iteraciones?          ", np.all(err <= prodZ + 1e-12))
print("¿prod Z_i <= exp(-2 sum gamma^2) en todas las iteraciones?", np.all(prodZ <= cota + 1e-12))
K0 = int(np.argmax(err == 0)) + 1 if np.any(err == 0) else None
print(f"error de entrenamiento final = {err[-1]:.3f}; es 0 desde K = {K0}  (P_i en [{min(hist['P']):.3f}, {max(hist['P']):.3f}])")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
it = np.arange(1, K + 1)
plt.figure(figsize=(7, 4))
plt.semilogy(it, np.where(err > 0, err, np.nan), label="error de entrenamiento (0-1)")
if K0: plt.axvline(K0, color="gray", ls=":", label=f"error = 0 desde K = {K0}")
plt.semilogy(it, prodZ, label=r"$\prod_{i\leq K} Z_i$  (pérdida exponencial media)")
plt.semilogy(it, cota, "--", label=r"$\exp(-2\sum_i\gamma_i^2)$")
plt.xlabel("iteración $K$"); plt.legend(); plt.title("Teorema de Freund y Schapire"); plt.tight_layout(); plt.show()
```

Todas las identidades de la derivación se cumplen hasta la precisión de máquina. El error de entrenamiento queda siempre por debajo de $\prod_i Z_i$, que decae geométricamente con $K$, y llega a **cero** en un número finito de iteraciones, como predice el Paso 5 del {prf:ref}`thm-fs`. La pendiente de la cota se suaviza a medida que los $P_i$ se acercan a $1/2$ (márgenes $\gamma_i$ más pequeños).
