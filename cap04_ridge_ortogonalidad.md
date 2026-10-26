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

# 4. Ridge: verificación de $Z^{T}\boldsymbol{1}=\boldsymbol{1}^{T}Z=0$

## Enunciado

```{figure} images/enunciado_ridge_Z1.png
:width: 95%

Enunciado: "Nótese que $Z^{T}\boldsymbol{1}=\boldsymbol{1}^{T}Z=0$ (verifíquelo)".
```

**Contexto (notas, Ecuaciones 3.4–3.7).** Con $\bar x_{j}=n^{-1}\sum_{i=1}^{n}x_{ij}$ y $s_{j}^2=n^{-1}\sum_{i=1}^{n}(x_{ij}-\bar x_{j})^{2}$, la matriz $Z\in\mathbb{R}^{n\times p}$ tiene entradas

$$
z_{ij}=\frac{x_{ij}-\bar x_{j}}{s_{j}},\qquad i=1,\dots,n,\ \ j=1,\dots,p,
$$ (eq-zij)

el modelo es $\boldsymbol{y}=\beta_{0}^{\star}\boldsymbol{1}+Z\boldsymbol{\beta}_{s}+\boldsymbol{\varepsilon}$, y el criterio ridge es

$$
S_{\lambda}(\beta_{0}^{\star}, \boldsymbol{\beta}_{s})=(\boldsymbol{y}-\beta_{0}^{\star}\boldsymbol{1}-Z\boldsymbol{\beta}_{s})^{T}(\boldsymbol{y}-\beta_{0}^{\star}\boldsymbol{1}-Z\boldsymbol{\beta}_{s})+\lambda\boldsymbol{\beta}_{s}^{T}\boldsymbol{\beta}_{s}.
$$ (eq-37)

Se supone $s_j>0$ para todo $j$ (ninguna variable predictora es constante); de lo contrario $z_{ij}$ no está definido.

## 4.1 Demostración

```{prf:proposition} Las columnas de $Z$ son ortogonales a $\boldsymbol{1}$
:label: prop-Z1

$$
Z^{T}\boldsymbol{1}=\boldsymbol{0}_p\in\mathbb{R}^{p}\qquad\text{y}\qquad \boldsymbol{1}^{T}Z=\boldsymbol{0}_p^{T}\in\mathbb{R}^{1\times p}.
$$
```

```{prf:proof}
*Paso 1: componente $j$ de $Z^{T}\boldsymbol{1}$.* La fila $j$ de $Z^{T}$ es la columna $j$ de $Z$, $\boldsymbol{z}_j=(z_{1j},\dots,z_{nj})^{T}$. Por lo tanto,

$$
(Z^{T}\boldsymbol{1})_j=\boldsymbol{z}_j^{T}\boldsymbol{1}=\sum_{i=1}^{n}z_{ij}\cdot1=\sum_{i=1}^{n}\frac{x_{ij}-\bar x_j}{s_j}.
$$

*Paso 2: sacar la constante.* $s_j$ no depende de $i$, así que

$$
\sum_{i=1}^{n}\frac{x_{ij}-\bar x_j}{s_j}=\frac{1}{s_j}\Big(\sum_{i=1}^{n}x_{ij}-\sum_{i=1}^{n}\bar x_j\Big)=\frac{1}{s_j}\big(n\bar x_j-n\bar x_j\big)=0,
$$

donde se usó $\sum_{i}x_{ij}=n\bar x_j$ (definición de la media) y $\sum_{i}\bar x_j=n\bar x_j$ (se suma $n$ veces la misma constante).

*Paso 3: conclusión.* Todas las componentes son cero, luego $Z^{T}\boldsymbol{1}=\boldsymbol{0}_p$. Trasponiendo, $\boldsymbol{1}^{T}Z=(Z^{T}\boldsymbol{1})^{T}=\boldsymbol{0}_p^{T}$.
```

```{admonition} Precisión sobre la notación
:class: note
La igualdad de las notas "$Z^{T}\boldsymbol{1}=\boldsymbol{1}^{T}Z=0$" se lee "ambos son nulos": $Z^{T}\boldsymbol{1}$ es un vector **columna** de $\mathbb{R}^p$ y $\boldsymbol{1}^{T}Z$ un vector **fila**, traspuesto del anterior. Geométricamente, el vector de unos es ortogonal a cada columna de $Z$, porque cada columna está centrada. La división por $s_j$ no interviene: basta con centrar.
```

## 4.2 Uso en las ecuaciones normales (3.8) y (3.10)

La proposición es justamente lo que permite **desacoplar** el intercepto de los demás coeficientes.

```{prf:corollary} Derivadas de $S_\lambda$ y estimadores ridge
:label: cor-normales

Las derivadas parciales de {eq}`eq-37` son

$$
\frac{\partial S_{\lambda}}{\partial\beta_{0}^{\star}}=-2n\overline{y}+2n\beta_{0}^{\star},\qquad
\frac{\partial S_{\lambda}}{\partial\boldsymbol{\beta}_{s}}=-2Z^{T}\boldsymbol{y}+2Z^{T}Z\boldsymbol{\beta}_{s}+2\lambda\boldsymbol{\beta}_{s},
$$

es decir, (3.8) y (3.10). El único punto crítico, $\hat{\beta}_{0}^{\star}=\overline{y}$ y $\hat{\boldsymbol{\beta}}_{s}=(Z^{T}Z+\lambda\boldsymbol{I}_{p})^{-1}Z^{T}\boldsymbol{y}$, es el mínimo global.
```

```{prf:proof}
*Desarrollo de $S_\lambda$.* Sea $\boldsymbol{r}=\boldsymbol{y}-\beta_0^{\star}\boldsymbol{1}-Z\boldsymbol{\beta}_s$. Desarrollando $\boldsymbol{r}^{T}\boldsymbol{r}$ y usando que cada producto escalar es igual a su traspuesto ($\boldsymbol{a}^{T}\boldsymbol{c}=\boldsymbol{c}^{T}\boldsymbol{a}$), junto con $\boldsymbol{1}^{T}\boldsymbol{1}=n$:

$$
S_{\lambda}=\boldsymbol{y}^{T}\boldsymbol{y}-2\beta_{0}^{\star}\boldsymbol{1}^{T}\boldsymbol{y}-2\boldsymbol{y}^{T}Z\boldsymbol{\beta}_{s}+2\beta_{0}^{\star}\,\boldsymbol{1}^{T}Z\,\boldsymbol{\beta}_{s}+n\beta_{0}^{\star2}+\boldsymbol{\beta}_{s}^{T}Z^{T}Z\boldsymbol{\beta}_{s}+\lambda\boldsymbol{\beta}_{s}^{T}\boldsymbol{\beta}_{s}.
$$

(En las notas, el término cruzado aparece como $2\beta_0\boldsymbol{1}^{T}Z\boldsymbol{\beta}_s$; debe leerse $\beta_0^{\star}$.)

*Derivada respecto de $\beta_0^{\star}$.* Con $\boldsymbol{1}^{T}\boldsymbol{y}=\sum_i y_i=n\overline{y}$,

$$
\frac{\partial S_{\lambda}}{\partial\beta_{0}^{\star}}=-2n\overline{y}+2\,\underbrace{\boldsymbol{1}^{T}Z}_{=\,\boldsymbol{0}^{T}}\boldsymbol{\beta}_{s}+2n\beta_{0}^{\star}=-2n\overline{y}+2n\beta_{0}^{\star}.
$$

Sin la {prf:ref}`prop-Z1` quedaría el término $2\boldsymbol{1}^{T}Z\boldsymbol{\beta}_s$ y la ecuación (3.8) dependería de $\boldsymbol{\beta}_s$.

*Derivada respecto de $\boldsymbol{\beta}_s$.* Usamos $\partial(\boldsymbol{a}^{T}\boldsymbol{\beta}_s)/\partial\boldsymbol{\beta}_s=\boldsymbol{a}$ y, para $A$ simétrica, $\partial(\boldsymbol{\beta}_s^{T}A\boldsymbol{\beta}_s)/\partial\boldsymbol{\beta}_s=2A\boldsymbol{\beta}_s$ (Ecuación 3.9, con $A=Z^{T}Z$ y $A=\boldsymbol{I}_p$):

$$
\frac{\partial S_{\lambda}}{\partial\boldsymbol{\beta}_{s}}=-2Z^{T}\boldsymbol{y}+2\beta_0^{\star}\underbrace{Z^{T}\boldsymbol{1}}_{=\,\boldsymbol{0}}+2Z^{T}Z\boldsymbol{\beta}_{s}+2\lambda\boldsymbol{\beta}_{s}=-2Z^{T}\boldsymbol{y}+2Z^{T}Z\boldsymbol{\beta}_{s}+2\lambda\boldsymbol{\beta}_{s}.
$$

*Solución.* Igualando a cero: $\hat\beta_0^{\star}=\overline{y}$ y $(Z^{T}Z+\lambda\boldsymbol{I}_p)\hat{\boldsymbol{\beta}}_s=Z^{T}\boldsymbol{y}$. La matriz es invertible para $\lambda>0$ ({prf:ref}`lem-Mlambda` con $Z$ en lugar de $X$).

*Es un mínimo.* La matriz hessiana de $S_\lambda$ respecto de $(\beta_0^{\star},\boldsymbol{\beta}_s)$ es

$$
H=2\begin{pmatrix} n & \boldsymbol{1}^{T}Z\\ Z^{T}\boldsymbol{1} & Z^{T}Z+\lambda\boldsymbol{I}_p\end{pmatrix}
=2\begin{pmatrix} n & \boldsymbol{0}^{T}\\ \boldsymbol{0} & Z^{T}Z+\lambda\boldsymbol{I}_p\end{pmatrix},
$$

diagonal por bloques (otra vez gracias a la {prf:ref}`prop-Z1`), con bloques definidos positivos. Por lo tanto $S_\lambda$ es estrictamente convexa y el punto crítico es su mínimo global único.
```

```{admonition} Consecuencia
:class: tip
Como $Z^{T}\boldsymbol{1}=\boldsymbol{0}$, se tiene $Z^{T}\boldsymbol{y}=Z^{T}(\boldsymbol{y}-\overline{y}\boldsymbol{1})$. Es decir, $\hat{\boldsymbol{\beta}}_s$ no cambia si se centra la respuesta. Esto justifica trabajar con $\boldsymbol{y}-\overline{y}\boldsymbol{1}$ y con el criterio reducido $S_\lambda(\boldsymbol{\beta}_s)=(\boldsymbol{y}-Z\boldsymbol{\beta}_s)^{T}(\boldsymbol{y}-Z\boldsymbol{\beta}_s)+\lambda\boldsymbol{\beta}_s^{T}\boldsymbol{\beta}_s$, como hacen las notas.
```

## 4.3 Verificación numérica

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng(7)
n, p, lam = 30, 5, 2.0
Xraw = rng.normal(loc=[10, -3, 0, 100, 5], scale=[1, 4, .2, 30, 2], size=(n, p))
y = 4 + Xraw @ np.array([1., -2., 3., .05, .5]) + rng.normal(size=n)

xbar, s = Xraw.mean(0), Xraw.std(0)          # s_j con divisor n, como en las notas
Z = (Xraw - xbar) / s
uno = np.ones(n)
print("Z^T 1 =", Z.T @ uno)
print("1^T Z =", uno @ Z)

# minimización conjunta de S_lambda(beta0*, beta_s) vs. fórmulas cerradas
A = np.column_stack([uno, Z])
P = np.diag([0] + [lam]*p)                    # el intercepto no se penaliza
sol = np.linalg.solve(A.T @ A + P, A.T @ y)
beta_s = np.linalg.solve(Z.T @ Z + lam*np.eye(p), Z.T @ y)
print("\nbeta0* (conjunto) =", sol[0].round(6), "  ybar =", y.mean().round(6))
print("máx |beta_s(conjunto) - (Z^TZ+lam I)^{-1} Z^T y| =", np.abs(sol[1:] - beta_s).max())
print("máx |Z^T y - Z^T (y - ybar 1)| =", np.abs(Z.T @ y - Z.T @ (y - y.mean())).max())
```
