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

# 3. Ridge: propiedades del estimador (3.12)

## Enunciado

```{figure} images/enunciado_ridge_props.png
:width: 95%

Enunciado: Propiedades del estimador ridge, Ecuación (3.12).
```

**Modelo.** Como en las notas, tras estandarizar (Capítulo 5) se considera, sin pérdida de generalidad,

$$
\begin{gathered}
\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon},\\
X\in\mathbb{R}^{n\times p}\ \text{(fija, no aleatoria)},\quad \boldsymbol{\beta}\in\mathbb{R}^{p},\quad \mathrm{E}(\boldsymbol{\varepsilon})=\boldsymbol{0},\quad \operatorname{Cov}(\boldsymbol{\varepsilon})=\sigma^{2}\boldsymbol{I}_n ,
\end{gathered}
$$

y el estimador ridge, con $\lambda>0$,

$$
\hat{\boldsymbol{\beta}}_{R}=(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}X^{T}\boldsymbol{y}.
$$

**Por demostrar (3.12):**

$$
\begin{aligned}
\text{(a)}\quad & \mathrm{E}(\hat{\boldsymbol{\beta}}_{R})=(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}X^{T}X\boldsymbol{\beta}\\
\text{(b)}\quad & \operatorname{Cov}(\hat{\boldsymbol{\beta}}_{R})=\sigma^{2}(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}X^{T}X(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}\\
\text{(c)}\quad & \mathrm{E}(\hat{\boldsymbol{\beta}}_{R}-\boldsymbol{\beta})=-\lambda(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}\boldsymbol{\beta}\\
\text{(d)}\quad & \mathrm{E}[(\hat{\boldsymbol{\beta}}_{R}-\boldsymbol{\beta})^{T}(\hat{\boldsymbol{\beta}}_{R}-\boldsymbol{\beta})]=\sigma^{2}\sum_{j=1}^{p}\frac{l_{j}}{(l_{j}+\lambda)^{2}}+\lambda^{2}\boldsymbol{\beta}^{T}(X^{T}X+\lambda\boldsymbol{I}_{p})^{-2}\boldsymbol{\beta},
\end{aligned}
$$

donde $l_1,\dots,l_p$ son los autovalores de $X^{T}X$.

```{admonition} Notación
:class: note
Para abreviar escribimos $\ M_\lambda:=X^{T}X+\lambda\boldsymbol{I}_p$. En el enunciado, la segunda línea dice $\operatorname{Cov}(\boldsymbol{\beta}_R)$; se entiende $\operatorname{Cov}(\hat{\boldsymbol{\beta}}_R)$, ya que $\boldsymbol{\beta}$ es un parámetro fijo y su covarianza sería nula.
```

## 3.1 Resultados auxiliares

```{prf:lemma} Propiedades de $M_\lambda$
:label: lem-Mlambda

Sea $X^{T}X=Q\Lambda Q^{T}$ su descomposición espectral, con $Q$ ortogonal ($Q^{T}Q=QQ^{T}=\boldsymbol{I}_p$) y $\Lambda=\operatorname{diag}(l_1,\dots,l_p)$. Entonces:

1. $l_j\ge0$ para todo $j$.
2. $M_\lambda=Q(\Lambda+\lambda\boldsymbol{I}_p)Q^{T}$ es simétrica y definida positiva; en particular, es invertible para todo $\lambda>0$, con $M_\lambda^{-1}=Q(\Lambda+\lambda\boldsymbol{I}_p)^{-1}Q^{T}$, también simétrica.
3. $M_\lambda^{-1}$ conmuta con $X^{T}X$: $\ M_\lambda^{-1}X^{T}X=X^{T}XM_\lambda^{-1}$.
```

```{prf:proof}
1. $X^{T}X$ es simétrica, así que el teorema espectral da $Q$ y $\Lambda$ reales. Si $X^{T}X\boldsymbol{q}=l\boldsymbol{q}$ con $\|\boldsymbol{q}\|=1$, entonces $l=\boldsymbol{q}^{T}X^{T}X\boldsymbol{q}=\|X\boldsymbol{q}\|^{2}\ge0$.
2. $M_\lambda=Q\Lambda Q^{T}+\lambda QQ^{T}=Q(\Lambda+\lambda\boldsymbol{I}_p)Q^{T}$. Sus autovalores son $l_j+\lambda\ge\lambda>0$, luego es definida positiva e invertible. Se verifica directamente que $Q(\Lambda+\lambda \boldsymbol{I}_p)^{-1}Q^{T}$ es su inversa: $Q(\Lambda+\lambda\boldsymbol{I}_p)Q^{T}\,Q(\Lambda+\lambda\boldsymbol{I}_p)^{-1}Q^{T}=Q\,\boldsymbol{I}_p\,Q^{T}=\boldsymbol{I}_p$. Es simétrica porque $(QDQ^{T})^{T}=QDQ^{T}$ para $D$ diagonal.
3. $M_\lambda^{-1}X^{T}X=Q(\Lambda+\lambda\boldsymbol{I})^{-1}Q^{T}Q\Lambda Q^{T}=Q(\Lambda+\lambda\boldsymbol{I})^{-1}\Lambda Q^{T}$, y $X^{T}XM_\lambda^{-1}=Q\Lambda(\Lambda+\lambda\boldsymbol{I})^{-1}Q^{T}$. Ambas coinciden porque las matrices diagonales conmutan.
```

```{prf:lemma} Esperanza y covarianza de transformaciones lineales
:label: lem-lineal

Si $\boldsymbol{u}$ es un vector aleatorio con $\mathrm{E}(\boldsymbol{u})=\boldsymbol{\mu}$ y $\operatorname{Cov}(\boldsymbol{u})=\Sigma$, y $A$, $\boldsymbol{b}$ son constantes, entonces $\mathrm{E}(A\boldsymbol{u}+\boldsymbol{b})=A\boldsymbol{\mu}+\boldsymbol{b}$ y $\operatorname{Cov}(A\boldsymbol{u}+\boldsymbol{b})=A\Sigma A^{T}$.
```

```{prf:proof}
La primera igualdad es la linealidad de la esperanza componente a componente: $\mathrm{E}\big[\sum_k A_{ik}u_k+b_i\big]=\sum_kA_{ik}\mu_k+b_i$. Para la segunda,

$$
\begin{aligned}
\operatorname{Cov}(A\boldsymbol{u}+\boldsymbol{b})&=\mathrm{E}\big[(A\boldsymbol{u}-A\boldsymbol{\mu})(A\boldsymbol{u}-A\boldsymbol{\mu})^{T}\big]\\
&=A\,\mathrm{E}\big[(\boldsymbol{u}-\boldsymbol{\mu})(\boldsymbol{u}-\boldsymbol{\mu})^{T}\big]A^{T}=A\Sigma A^{T}.
\end{aligned}
$$
```

```{prf:lemma} Error cuadrático = traza de la covarianza + sesgo al cuadrado
:label: lem-ecm

Si $\hat{\boldsymbol{\theta}}$ estima $\boldsymbol{\theta}$ y $\boldsymbol{b}:=\mathrm{E}(\hat{\boldsymbol{\theta}})-\boldsymbol{\theta}$, entonces

$$
\mathrm{E}\big[(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})^{T}(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta})\big]=\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol{\theta}})+\boldsymbol{b}^{T}\boldsymbol{b}.
$$
```

```{prf:proof}
Sea $\boldsymbol{e}=\hat{\boldsymbol{\theta}}-\boldsymbol{\theta}$. Un escalar es su propia traza, y $\operatorname{tr}(\boldsymbol{a}^{T}\boldsymbol{c})=\operatorname{tr}(\boldsymbol{c}\boldsymbol{a}^{T})$, así que $\boldsymbol{e}^{T}\boldsymbol{e}=\operatorname{tr}(\boldsymbol{e}\boldsymbol{e}^{T})$. La traza es lineal y conmuta con la esperanza:

$$
\mathrm{E}[\boldsymbol{e}^{T}\boldsymbol{e}]=\operatorname{tr}\mathrm{E}[\boldsymbol{e}\boldsymbol{e}^{T}].
$$

Ahora $\mathrm{E}(\boldsymbol{e})=\boldsymbol{b}$ y $\operatorname{Cov}(\boldsymbol{e})=\operatorname{Cov}(\hat{\boldsymbol{\theta}})$, porque restar la constante $\boldsymbol{\theta}$ no cambia la covarianza. Usando $\mathrm{E}[\boldsymbol{e}\boldsymbol{e}^{T}]=\operatorname{Cov}(\boldsymbol{e})+\mathrm{E}(\boldsymbol{e})\mathrm{E}(\boldsymbol{e})^{T}$,

$$
\mathrm{E}[\boldsymbol{e}^{T}\boldsymbol{e}]=\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol{\theta}})+\operatorname{tr}(\boldsymbol{b}\boldsymbol{b}^{T})=\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol{\theta}})+\boldsymbol{b}^{T}\boldsymbol{b}.
$$
```

Como $X\boldsymbol{\beta}$ es constante, el {prf:ref}`lem-lineal` da

$$
\mathrm{E}(\boldsymbol{y})=X\boldsymbol{\beta},\qquad \operatorname{Cov}(\boldsymbol{y})=\operatorname{Cov}(\boldsymbol{\varepsilon})=\sigma^{2}\boldsymbol{I}_n.
$$ (eq-momentos-y)

Además, $\hat{\boldsymbol{\beta}}_R=A\boldsymbol{y}$ con $A:=M_\lambda^{-1}X^{T}$ (matriz constante $p\times n$).

## 3.2 Demostración de (3.12)

```{prf:theorem} Propiedades (3.12) del estimador ridge
:label: thm-ridge-props

Bajo el modelo anterior y $\lambda>0$, se cumplen (a), (b), (c) y (d).
```

```{prf:proof}
**(a) Esperanza.** Por el {prf:ref}`lem-lineal` y {eq}`eq-momentos-y`,

$$
\mathrm{E}(\hat{\boldsymbol{\beta}}_R)=M_\lambda^{-1}X^{T}\,\mathrm{E}(\boldsymbol{y})=M_\lambda^{-1}X^{T}X\boldsymbol{\beta}=(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}X^{T}X\boldsymbol{\beta}.
$$

**(b) Covarianza.** De nuevo por el {prf:ref}`lem-lineal`, con $A=M_\lambda^{-1}X^{T}$ y $A^{T}=X(M_\lambda^{-1})^{T}=XM_\lambda^{-1}$ (por la simetría del {prf:ref}`lem-Mlambda`),

$$
\begin{aligned}
\operatorname{Cov}(\hat{\boldsymbol{\beta}}_R)&=A\,(\sigma^{2}\boldsymbol{I}_n)\,A^{T}=\sigma^{2}M_\lambda^{-1}X^{T}XM_\lambda^{-1}\\
&=\sigma^{2}(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}X^{T}X(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}.
\end{aligned}
$$

**(c) Sesgo.** Usando (a) y escribiendo $\boldsymbol{\beta}=M_\lambda^{-1}M_\lambda\boldsymbol{\beta}$,

$$
\begin{aligned}
\mathrm{E}(\hat{\boldsymbol{\beta}}_R-\boldsymbol{\beta})&=M_\lambda^{-1}X^{T}X\boldsymbol{\beta}-M_\lambda^{-1}M_\lambda\boldsymbol{\beta}
=M_\lambda^{-1}\big(X^{T}X-X^{T}X-\lambda\boldsymbol{I}_p\big)\boldsymbol{\beta}\\
&=-\lambda(X^{T}X+\lambda\boldsymbol{I}_{p})^{-1}\boldsymbol{\beta}.
\end{aligned}
$$

**(d) Error cuadrático medio.** Por el {prf:ref}`lem-ecm`,

$$
\mathrm{E}[(\hat{\boldsymbol{\beta}}_{R}-\boldsymbol{\beta})^{T}(\hat{\boldsymbol{\beta}}_{R}-\boldsymbol{\beta})]=\underbrace{\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol{\beta}}_R)}_{\text{(I)}}+\underbrace{\boldsymbol{b}^{T}\boldsymbol{b}}_{\text{(II)}},\qquad \boldsymbol{b}=-\lambda M_\lambda^{-1}\boldsymbol{\beta}.
$$

*Término (I).* Por (b) y el {prf:ref}`lem-Mlambda`,

$$
\begin{aligned}
M_\lambda^{-1}X^{T}XM_\lambda^{-1}&=Q(\Lambda+\lambda\boldsymbol{I})^{-1}Q^{T}\,Q\Lambda Q^{T}\,Q(\Lambda+\lambda\boldsymbol{I})^{-1}Q^{T}\\
&=Q\,\operatorname{diag}\!\left(\frac{l_1}{(l_1+\lambda)^2},\dots,\frac{l_p}{(l_p+\lambda)^2}\right)Q^{T}.
\end{aligned}
$$

Como $\operatorname{tr}(QDQ^{T})=\operatorname{tr}(DQ^{T}Q)=\operatorname{tr}(D)$ (la traza es invariante bajo permutaciones cíclicas),

$$
\text{(I)}=\sigma^{2}\operatorname{tr}\big(M_\lambda^{-1}X^{T}XM_\lambda^{-1}\big)=\sigma^{2}\sum_{j=1}^{p}\frac{l_j}{(l_j+\lambda)^{2}}.
$$

*Término (II).* Como $M_\lambda^{-1}$ es simétrica, $(M_\lambda^{-1}\boldsymbol{\beta})^{T}=\boldsymbol{\beta}^{T}M_\lambda^{-1}$ y

$$
\text{(II)}=(-\lambda M_\lambda^{-1}\boldsymbol{\beta})^{T}(-\lambda M_\lambda^{-1}\boldsymbol{\beta})=\lambda^{2}\boldsymbol{\beta}^{T}M_\lambda^{-1}M_\lambda^{-1}\boldsymbol{\beta}=\lambda^{2}\boldsymbol{\beta}^{T}(X^{T}X+\lambda\boldsymbol{I}_{p})^{-2}\boldsymbol{\beta}.
$$

Sumando (I) y (II) se obtiene (d).
```

```{admonition} Interpretación de (d)
:class: tip
Los elementos diagonales de $\operatorname{Cov}(\hat{\boldsymbol{\beta}}_R)$ son las varianzas $\operatorname{Var}(\hat{\beta}_{R,j})$; su suma es la traza. Por eso el primer término de (d) es la **suma de las varianzas** de las componentes y el segundo es el **cuadrado del sesgo** $\|\mathrm{E}(\hat{\boldsymbol{\beta}}_R)-\boldsymbol{\beta}\|^{2}$, como dicen las notas. Escribiendo $\alpha_j:=\boldsymbol{q}_j^{T}\boldsymbol{\beta}$ (coordenadas de $\boldsymbol{\beta}$ en la base de autovectores), el segundo término también es $\lambda^{2}\sum_{j}\alpha_j^{2}/(l_j+\lambda)^{2}$.
```

## 3.3 Consecuencia: ridge mejora a mínimos cuadrados para algún $\lambda>0$

```{prf:corollary} Hoerl y Kennard (1970)
:label: cor-hoerl

Supongamos que $X^{T}X$ es invertible ($l_j>0$ para todo $j$), y sea $\operatorname{ECM}(\lambda)$ el lado derecho de (d). Entonces $\operatorname{ECM}(0)=\sigma^{2}\sum_j 1/l_j$ es el error del estimador de mínimos cuadrados, y

$$
\operatorname{ECM}'(\lambda)=2\sum_{j=1}^{p}\frac{l_j\,(\lambda\alpha_j^{2}-\sigma^{2})}{(l_j+\lambda)^{3}} .
$$

En particular, $\operatorname{ECM}'(\lambda)<0$ para $0\le\lambda<\sigma^{2}/\max_j\alpha_j^{2}$. Por lo tanto existe $\lambda>0$ con $\operatorname{ECM}(\lambda)<\operatorname{ECM}(0)$.
```

```{prf:proof}
Por la interpretación anterior, $\operatorname{ECM}(\lambda)=\sum_j\big[\sigma^{2}l_j+\lambda^{2}\alpha_j^{2}\big]/(l_j+\lambda)^{2}$. Derivando cada sumando,

$$
\begin{gathered}
\frac{d}{d\lambda}\frac{\sigma^{2}l_j}{(l_j+\lambda)^{2}}=-\frac{2\sigma^{2}l_j}{(l_j+\lambda)^{3}},\\
\frac{d}{d\lambda}\frac{\lambda^{2}\alpha_j^{2}}{(l_j+\lambda)^{2}}=\alpha_j^{2}\,\frac{2\lambda(l_j+\lambda)-2\lambda^{2}}{(l_j+\lambda)^{3}}=\frac{2\lambda l_j\alpha_j^{2}}{(l_j+\lambda)^{3}}.
\end{gathered}
$$

Sumando se obtiene la fórmula. Si $\lambda\alpha_j^{2}<\sigma^{2}$ para todo $j$, cada sumando es negativo. Esto ocurre en el intervalo $[0,\sigma^{2}/\max_j\alpha_j^{2})$, que es no vacío (si todos los $\alpha_j$ son nulos, ocurre para todo $\lambda\ge 0$). Como $\operatorname{ECM}$ es estrictamente decreciente en ese intervalo, cualquier $\lambda$ en su interior cumple $\operatorname{ECM}(\lambda)<\operatorname{ECM}(0)$.
```

## 3.4 Verificación numérica (Monte Carlo)

Se fija una matriz $X$ con columnas muy correlacionadas (multicolinealidad), se simulan $200\,000$ réplicas de $\boldsymbol{\varepsilon}$ y se comparan las cuatro fórmulas de (3.12) con sus estimaciones empíricas.

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng(42)
n, p, sigma, lam, R = 40, 4, 1.5, 3.0, 200_000

Z0 = rng.normal(size=(n, 1))
X = Z0 + 0.15*rng.normal(size=(n, p))              # columnas casi colineales
X = (X - X.mean(0)) / X.std(0)                     # estandarización (3.11)
beta = np.array([2.0, -1.0, 0.5, 1.0])

XtX = X.T @ X
M_inv = np.linalg.inv(XtX + lam*np.eye(p))
l = np.linalg.eigvalsh(XtX)

# fórmulas teóricas (3.12)
E_teo   = M_inv @ XtX @ beta
Cov_teo = sigma**2 * M_inv @ XtX @ M_inv
ses_teo = -lam * M_inv @ beta
ecm_teo = sigma**2*np.sum(l/(l+lam)**2) + lam**2 * beta @ M_inv @ M_inv @ beta

# Monte Carlo
Y = X @ beta + sigma*rng.normal(size=(R, n))       # R réplicas de y
B = Y @ X @ M_inv.T                                 # cada fila: beta_R = M^{-1} X^T y
E_mc, Cov_mc = B.mean(0), np.cov(B.T)
ecm_mc = np.mean(np.sum((B - beta)**2, axis=1))

np.set_printoptions(precision=4, suppress=True)
print("E(beta_R)   teórica:", E_teo, "\n            MC     :", E_mc)
print("sesgo       teórico:", ses_teo, "\n            MC     :", E_mc - beta)
print("máx |Cov_teo - Cov_mc| =", np.abs(Cov_teo - Cov_mc).max(), " (escala de Cov:", np.abs(Cov_teo).max().round(4), ")")
print(f"ECM teórico = {ecm_teo:.4f}   MC = {ecm_mc:.4f}")
print(f"tr Cov = {np.trace(Cov_teo):.4f} = sigma^2 sum l/(l+lam)^2 = {sigma**2*np.sum(l/(l+lam)**2):.4f}")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
l, Q = np.linalg.eigh(XtX); alpha = Q.T @ beta     # alpha_j = q_j^T beta
lams = np.linspace(0, 15, 400)
var = np.array([sigma**2*np.sum(l/(l+t)**2) for t in lams])
ses = np.array([t**2*np.sum(alpha**2/(l+t)**2) for t in lams])
plt.figure(figsize=(7, 4))
plt.plot(lams, var, label="suma de varianzas")
plt.plot(lams, ses, label="sesgo$^2$")
plt.plot(lams, var + ses, "k", lw=2, label="ECM (3.12)")
plt.axhline(var[0], ls=":", c="gray", label="ECM de MCO ($\\lambda=0$)")
plt.xlabel("$\\lambda$"); plt.legend(); plt.yscale("log"); plt.tight_layout(); plt.show()
print("lambda óptimo ≈", lams[np.argmin(var+ses)].round(3), " | ECM(0) =", var[0].round(3), " ECM mínimo =", (var+ses).min().round(3))
```

Las estimaciones de Monte Carlo coinciden con las fórmulas hasta el error de muestreo. La gráfica ilustra el {prf:ref}`cor-hoerl`: con multicolinealidad, $l_{\min}$ es pequeño, $\sigma^2/l_{\min}$ es enorme, y un $\lambda$ moderado reduce mucho el ECM.
