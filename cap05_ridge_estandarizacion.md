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

# 5. Ridge: verificación de las igualdades (3.11)

## Enunciado

```{figure} images/enunciado_ridge_std.png
:width: 95%

Enunciado: "Estandarizando las variables predictoras y respuesta [...] se tienen las siguientes igualdades, verifíquelas".
```

Las notas estandarizan los datos del modelo (3.3) mediante

$$
y_i\;\longmapsto\;y_i-\overline{y},\qquad x_{ij}\;\longmapsto\;\frac{x_{ij}-\bar x_j}{s_j},
$$

con $\overline{y}=n^{-1}\sum_i y_i$, $\bar x_{j}=n^{-1}\sum_{i=1}^{n}x_{ij}$ y $s_{j}^2=n^{-1}\sum_{i=1}^{n}(x_{ij}-\bar x_{j})^{2}$ (Ecuación 3.4, **con divisor $n$**). Después de transformar, se **reutiliza** la misma notación $y_i$, $x_{ij}$ para los datos estandarizados. Para que la demostración sea clara, aquí distinguimos con tilde:

$$
\tilde y_i:=y_i-\overline{y},\qquad \tilde x_{ij}:=\frac{x_{ij}-\bar x_j}{s_j}\quad(s_j>0).
$$

**Por demostrar (3.11):**

$$
\sum_{i=1}^{n}\tilde y_{i}=0,\qquad\sum_{i=1}^{n}\tilde x_{ij}=0,\ \ j=1,2,\dots,p,\qquad\sum_{i=1}^{n}\tilde x_{ij}^{2}=n,
$$

y, como consecuencia, $\beta_{0}^{\star}=\bar{\tilde y}=\frac1n\sum_{i=1}^{n}\tilde y_{i}=0$.

## 5.1 Demostración

```{prf:proposition} Igualdades (3.11)
:label: prop-311

Para datos estandarizados como arriba:

1. $\displaystyle\sum_{i=1}^{n}\tilde y_i=0$.
2. $\displaystyle\sum_{i=1}^{n}\tilde x_{ij}=0$ para cada $j=1,\dots,p$.
3. $\displaystyle\sum_{i=1}^{n}\tilde x_{ij}^{2}=n$ para cada $j=1,\dots,p$.
```

```{prf:proof}
**1.** Separando la suma y usando $\sum_{i=1}^{n}y_i=n\overline{y}$:

$$
\sum_{i=1}^{n}\tilde y_i=\sum_{i=1}^{n}(y_i-\overline{y})=\sum_{i=1}^{n}y_i-n\overline{y}=n\overline{y}-n\overline{y}=0.
$$

**2.** Con el mismo argumento, y sacando el factor constante $1/s_j$:

$$
\sum_{i=1}^{n}\tilde x_{ij}=\frac{1}{s_j}\sum_{i=1}^{n}(x_{ij}-\bar x_j)=\frac{1}{s_j}\big(n\bar x_j-n\bar x_j\big)=0.
$$

(Esta es exactamente la {prf:ref}`prop-Z1`: $\sum_i\tilde x_{ij}=(Z^{T}\boldsymbol{1})_j$.)

**3.** Elevando al cuadrado y sacando $1/s_j^{2}$:

$$
\sum_{i=1}^{n}\tilde x_{ij}^{2}=\sum_{i=1}^{n}\frac{(x_{ij}-\bar x_j)^{2}}{s_j^{2}}=\frac{1}{s_j^{2}}\sum_{i=1}^{n}(x_{ij}-\bar x_j)^{2}.
$$

Por la definición de $s_j^2$, $\sum_{i=1}^{n}(x_{ij}-\bar x_j)^{2}=n\,s_j^{2}$. Por lo tanto,

$$
\sum_{i=1}^{n}\tilde x_{ij}^{2}=\frac{n\,s_j^{2}}{s_j^{2}}=n.
$$
```

```{prf:corollary} El intercepto estandarizado es nulo
:label: cor-beta0

En el modelo con datos estandarizados, el estimador del intercepto del Capítulo 4, $\hat\beta_0^{\star}=\overline{\tilde y}$, vale

$$
\beta_{0}^{\star}=\overline{\tilde y}=\frac{1}{n}\sum_{i=1}^{n}\tilde y_{i}=\frac1n\cdot0=0 .
$$

Por eso las notas pueden trabajar, sin pérdida de generalidad, con el modelo sin intercepto $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$, donde $X$ es la matriz estandarizada.
```

```{prf:proof}
Por el {prf:ref}`cor-normales`, aplicado con la respuesta $\tilde{\boldsymbol{y}}$, el intercepto óptimo es la media de la respuesta. Por el punto 1 de la {prf:ref}`prop-311`, esa media es cero.
```

## 5.2 Consecuencias útiles

```{prf:corollary} La matriz $X^{T}X$ estandarizada
:label: cor-XtX-corr

Si $X=(\tilde x_{ij})$, entonces $(X^{T}X)_{jj}=n$ y $(X^{T}X)_{jk}=n\,r_{jk}$, donde $r_{jk}$ es el coeficiente de correlación de Pearson entre las variables $j$ y $k$. Es decir, $X^{T}X=n\,R$, con $R$ la matriz de correlaciones. En particular, $\operatorname{tr}(X^{T}X)=\sum_j l_j=np$.
```

```{prf:proof}
$(X^{T}X)_{jk}=\sum_i\tilde x_{ij}\tilde x_{ik}=\frac{1}{s_js_k}\sum_i(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)=n\,\frac{\widehat{\operatorname{cov}}(x_j,x_k)}{s_js_k}=n\,r_{jk}$, con la covarianza muestral de divisor $n$. Para $j=k$ se obtiene el punto 3 de la {prf:ref}`prop-311`. La traza es la suma de los autovalores.
```

```{admonition} Cuidado con el divisor
:class: warning
La igualdad $\sum_i\tilde x_{ij}^{2}=n$ depende de usar la varianza con divisor $n$, como hacen las notas. Si se usa la cuasivarianza $s_j^{2}=\frac{1}{n-1}\sum_i(x_{ij}-\bar x_j)^2$ (por ejemplo, `pandas.Series.std()` por defecto), se obtiene $\sum_i\tilde x_{ij}^{2}=n-1$. `numpy.std` (con `ddof=0`) y `sklearn.preprocessing.StandardScaler` usan divisor $n$, de modo que reproducen (3.11).
```

## 5.3 Verificación numérica

```{code-cell} ipython3
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

Xraw, y = load_diabetes(return_X_y=True, scaled=False)
n = len(y)
y_t = y - y.mean()
X_t = (Xraw - Xraw.mean(0)) / Xraw.std(0)          # divisor n

print("n =", n)
print("sum y~        =", y_t.sum().round(10))
print("sum x~_ij     =", X_t.sum(0).round(10))
print("sum x~_ij^2   =", (X_t**2).sum(0).round(10))
print("beta0* = mean(y~) =", y_t.mean().round(12))
print("¿StandardScaler reproduce (3.11)?", np.allclose((StandardScaler().fit_transform(Xraw)**2).sum(0), n))
print("Con cuasivarianza (pandas .std):", (((Xraw - Xraw.mean(0)) / pd.DataFrame(Xraw).std().values)**2).sum(0).round(6)[:3], "... = n-1 =", n-1)
print("¿X^T X = n R?", np.allclose(X_t.T @ X_t, n*np.corrcoef(Xraw.T)))
```
