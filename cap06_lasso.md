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

# 6. Lasso: estimación de los parámetros

## Enunciado

```{figure} images/enunciado_lasso_b.png
:width: 95%

Formulación lasso de las notas.
```

```{figure} images/enunciado_lasso_a.png
:width: 95%

Ejercicio para el lector: "encontrar parámetros de estimación lasso, tal como se realizó en el caso de la regresión ridge. Se sugiere investigar sobre el algoritmo LARS (Least Angle Regression) de Efron et al. (2004)".
```

El estimador lasso minimiza

$$
S_{\lambda}(\boldsymbol{\beta})=(\boldsymbol{y}-X\boldsymbol{\beta})^{T}(\boldsymbol{y}-X\boldsymbol{\beta})+\lambda\sum_{j=1}^{p}|\beta_{j}|,\qquad \lambda>0,
$$ (eq-lasso)

con datos normalizados como en (3.11) (Capítulo 5): $\sum_i y_i=0$, $\sum_i x_{ij}=0$ y $\sum_i x_{ij}^{2}=n$, es decir, $\boldsymbol{x}_j^{T}\boldsymbol{x}_j=n$ para cada columna $\boldsymbol{x}_j$ de $X$. Por el {prf:ref}`cor-beta0`, no hay intercepto. En las notas el índice de la suma aparece como $i$; aquí se escribe $j$, que es el índice de los coeficientes.

**Diferencia con ridge.** En ridge, $S_\lambda$ es diferenciable y basta con resolver $\nabla S_\lambda=\boldsymbol{0}$ (Capítulos 3–4), lo que da la fórmula cerrada $\hat{\boldsymbol{\beta}}_R=(X^TX+\lambda\boldsymbol{I}_p)^{-1}X^T\boldsymbol{y}$. En lasso, $|\beta_j|$ **no es derivable en $\beta_j=0$**, de modo que:

1. las ecuaciones normales se sustituyen por **condiciones de optimalidad con subgradientes** (Sección 6.2);
2. hay fórmula cerrada en el caso de diseño ortogonal: la **umbralización suave** (Sección 6.3);
3. en general, $\hat{\boldsymbol{\beta}}$ se calcula con algoritmos: **descenso por coordenadas** (Sección 6.4) y **LARS** (Sección 6.5), que además da todo el camino $\lambda\mapsto\hat{\boldsymbol{\beta}}(\lambda)$.

Notación: $\boldsymbol{c}(\boldsymbol{\beta}):=X^{T}(\boldsymbol{y}-X\boldsymbol{\beta})$, cuya componente $c_j=\boldsymbol{x}_j^{T}(\boldsymbol{y}-X\boldsymbol{\beta})$ es la **correlación** (sin normalizar) de la variable $j$ con el residuo; $\operatorname{sgn}(t)\in\{-1,0,1\}$ es la función signo; $(t)_+=\max\{t,0\}$.

## 6.1 Existencia y unicidad

```{prf:proposition} Existencia del estimador lasso
:label: prop-lasso-existe

1. $S_\lambda$ es convexa y continua, y para $\lambda>0$ tiene al menos un minimizador.
2. Si $\hat{\boldsymbol{\beta}}^{(1)}$ y $\hat{\boldsymbol{\beta}}^{(2)}$ son minimizadores, entonces $X\hat{\boldsymbol{\beta}}^{(1)}=X\hat{\boldsymbol{\beta}}^{(2)}$ (los valores ajustados son únicos) y $\|\hat{\boldsymbol{\beta}}^{(1)}\|_1=\|\hat{\boldsymbol{\beta}}^{(2)}\|_1$.
3. Si $X$ tiene rango columna completo, el minimizador es único.
```

```{prf:proof}
1. $\boldsymbol{\beta}\mapsto\|\boldsymbol{y}-X\boldsymbol{\beta}\|^{2}$ es convexa (es la composición de la norma al cuadrado, que es convexa, con una función afín), y cada $|\beta_j|$ es convexa. Una suma con pesos no negativos de funciones convexas es convexa. Además $S_\lambda(\boldsymbol{\beta})\ge\lambda\|\boldsymbol{\beta}\|_1$, así que el conjunto de nivel $\{S_\lambda\le S_\lambda(\boldsymbol{0})\}$ está contenido en la bola $\{\|\boldsymbol{\beta}\|_1\le\boldsymbol{y}^{T}\boldsymbol{y}/\lambda\}$. Ese conjunto es cerrado (por continuidad) y acotado, es decir, compacto. Por el teorema de Weierstrass, $S_\lambda$ alcanza su mínimo en él, y ese mínimo es global.
2. Sea $m$ el valor mínimo y $\bar{\boldsymbol{\beta}}=\frac12(\hat{\boldsymbol{\beta}}^{(1)}+\hat{\boldsymbol{\beta}}^{(2)})$. Si $X\hat{\boldsymbol{\beta}}^{(1)}\neq X\hat{\boldsymbol{\beta}}^{(2)}$, la convexidad **estricta** de $\boldsymbol{u}\mapsto\|\boldsymbol{y}-\boldsymbol{u}\|^2$ y la convexidad de $\|\cdot\|_1$ darían $S_\lambda(\bar{\boldsymbol{\beta}})<\frac12S_\lambda(\hat{\boldsymbol{\beta}}^{(1)})+\frac12S_\lambda(\hat{\boldsymbol{\beta}}^{(2)})=m$, una contradicción. Luego los ajustes coinciden, las sumas de cuadrados también, y como $S_\lambda$ es la misma, también $\lambda\|\cdot\|_1$.
3. Si $X$ tiene rango completo, $X\boldsymbol{\beta}=X\boldsymbol{\beta}'$ implica $\boldsymbol{\beta}=\boldsymbol{\beta}'$; se aplica el punto 2.
```

## 6.2 Condiciones de optimalidad (las "ecuaciones normales" del lasso)

El **subdiferencial** de $|t|$ es $\partial|t|=\{\operatorname{sgn}(t)\}$ si $t\neq0$ y $\partial|0|=[-1,1]$: es el conjunto de pendientes $v$ tales que $|s|\ge|t|+v(s-t)$ para todo $s$.

```{prf:theorem} Condiciones de Karush–Kuhn–Tucker del lasso
:label: thm-kkt

$\hat{\boldsymbol{\beta}}$ minimiza {eq}`eq-lasso` si y solo si, para cada $j=1,\dots,p$,

$$
\begin{gathered}
\boldsymbol{x}_j^{T}(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})=\frac{\lambda}{2}\operatorname{sgn}(\hat\beta_j)\quad\text{si }\hat\beta_j\neq0,\\
\big|\boldsymbol{x}_j^{T}(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})\big|\le\frac{\lambda}{2}\quad\text{si }\hat\beta_j=0.
\end{gathered}
$$ (eq-kkt)

Equivalentemente, $\ \boldsymbol{0}\in-2X^{T}(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})+\lambda\,\partial\|\hat{\boldsymbol{\beta}}\|_1$.
```

```{prf:proof}
Sea $c_j=\boldsymbol{x}_j^{T}(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})$.

**($\Rightarrow$) Necesidad.** Fijemos $j$ y movamos solo la coordenada $j$: $\varphi(t):=S_\lambda(\hat{\boldsymbol{\beta}}+t\boldsymbol{e}_j)$, que tiene un mínimo en $t=0$. Como $\boldsymbol{y}-X(\hat{\boldsymbol{\beta}}+t\boldsymbol{e}_j)=(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})-t\boldsymbol{x}_j$ y $\boldsymbol{x}_j^{T}\boldsymbol{x}_j=n$,

$$
\varphi(t)-\varphi(0)=-2tc_j+nt^{2}+\lambda\big(|\hat\beta_j+t|-|\hat\beta_j|\big)\ \ge\ 0\quad\forall t.
$$

- Si $\hat\beta_j\neq0$: $\varphi$ es derivable en $0$ y $\varphi'(0)=-2c_j+\lambda\operatorname{sgn}(\hat\beta_j)=0$, de donde $c_j=\frac{\lambda}{2}\operatorname{sgn}(\hat\beta_j)$.
- Si $\hat\beta_j=0$: $-2tc_j+nt^2+\lambda|t|\ge0$. Para $t>0$, dividiendo por $t$ y haciendo $t\to0^+$, se obtiene $\lambda-2c_j\ge0$. Para $t<0$, dividiendo por $|t|$ y haciendo $t\to0^-$, se obtiene $\lambda+2c_j\ge0$. Juntas: $|c_j|\le\lambda/2$.

**($\Leftarrow$) Suficiencia.** Supongamos {eq}`eq-kkt` y definamos $v_j:=2c_j/\lambda$. Entonces $|v_j|\le1$ para todo $j$, y $v_j=\operatorname{sgn}(\hat\beta_j)$ si $\hat\beta_j\neq0$; en ambos casos $|\hat\beta_j|=v_j\hat\beta_j$. Sea $\boldsymbol{\beta}$ arbitrario.

- Parte cuadrática: como $Q(\boldsymbol{\beta})=\|\boldsymbol{y}-X\boldsymbol{\beta}\|^{2}$ es convexa y derivable, está por encima de su tangente: $Q(\boldsymbol{\beta})\ge Q(\hat{\boldsymbol{\beta}})-2(\boldsymbol{y}-X\hat{\boldsymbol{\beta}})^{T}X(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})=Q(\hat{\boldsymbol{\beta}})-2\boldsymbol{c}^{T}(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})$.
- Parte $L^1$: $|\beta_j|\ge v_j\beta_j=|\hat\beta_j|+v_j(\beta_j-\hat\beta_j)$, porque $|v_j|\le1$.

Sumando, con $\lambda\boldsymbol{v}=2\boldsymbol{c}$:

$$
S_\lambda(\boldsymbol{\beta})\ \ge\ S_\lambda(\hat{\boldsymbol{\beta}})+\big(-2\boldsymbol{c}+\lambda\boldsymbol{v}\big)^{T}(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}})=S_\lambda(\hat{\boldsymbol{\beta}}).
$$

Así $\hat{\boldsymbol{\beta}}$ es un mínimo global.
```

```{admonition} Lectura de las condiciones (3.8)/(3.10) del lasso
:class: tip
- Las variables **activas** ($\hat\beta_j\neq0$) tienen todas la **misma** correlación absoluta con el residuo, $|c_j|=\lambda/2$, y su signo coincide con el del coeficiente.
- Las variables **inactivas** tienen correlación a lo sumo $\lambda/2$.
- Comparando con ridge, donde $\boldsymbol{c}=\lambda\boldsymbol{\beta}$ ({prf:ref}`cor-normales`), en ridge la correlación es proporcional al coeficiente y $\hat\beta_j=0$ solo si $c_j=0$. En lasso, en cambio, hay toda una "zona muerta" $|c_j|\le\lambda/2$ donde el coeficiente es **exactamente cero**. Esa es la razón analítica de la selección de variables que las notas ilustran con la Fig. 3.2.
```

```{prf:corollary} Valor crítico $\lambda_{\max}$
:label: cor-lmax

$\hat{\boldsymbol{\beta}}=\boldsymbol{0}$ es solución si y solo si $\lambda\ge\lambda_{\max}:=2\max_j|\boldsymbol{x}_j^{T}\boldsymbol{y}|$.
```

```{prf:proof}
Con $\hat{\boldsymbol{\beta}}=\boldsymbol{0}$ todas las coordenadas son nulas, y {eq}`eq-kkt` se reduce a $|\boldsymbol{x}_j^{T}\boldsymbol{y}|\le\lambda/2$ para todo $j$.
```

## 6.3 Diseño ortogonal: fórmula cerrada (umbralización suave)

Para comparar "tal como se hizo con ridge", suponemos un diseño estandarizado y **ortogonal**, $X^{T}X=n\boldsymbol{I}_p$; es decir, (3.11) más variables incorreladas. En ese caso, el estimador de mínimos cuadrados es $\hat{\boldsymbol{\beta}}^{\text{MCO}}=(X^{T}X)^{-1}X^{T}\boldsymbol{y}=\frac1nX^{T}\boldsymbol{y}$.

```{prf:theorem} Lasso y ridge con diseño ortogonal
:label: thm-soft

Si $X^{T}X=n\boldsymbol{I}_p$, el estimador lasso es único y vale, componente a componente,

$$
\hat\beta_j^{L}=\mathcal{S}\!\left(\hat\beta_j^{\text{MCO}},\tfrac{\lambda}{2n}\right):=\operatorname{sgn}\big(\hat\beta_j^{\text{MCO}}\big)\left(\big|\hat\beta_j^{\text{MCO}}\big|-\frac{\lambda}{2n}\right)_{+},
$$ (eq-soft)

mientras que el estimador ridge es

$$
\hat{\boldsymbol{\beta}}_R=(n\boldsymbol{I}_p+\lambda\boldsymbol{I}_p)^{-1}X^{T}\boldsymbol{y}=\frac{n}{n+\lambda}\,\hat{\boldsymbol{\beta}}^{\text{MCO}}.
$$
```

```{prf:proof}
*Ridge:* es inmediato, sustituyendo $X^{T}X=n\boldsymbol{I}_p$ en $(X^{T}X+\lambda\boldsymbol{I}_p)^{-1}X^{T}\boldsymbol{y}$.

*Lasso.* $X$ tiene rango completo, así que el minimizador es único ({prf:ref}`prop-lasso-existe`). Bajo ortogonalidad, $c_j=\boldsymbol{x}_j^{T}\boldsymbol{y}-\boldsymbol{x}_j^{T}X\hat{\boldsymbol{\beta}}=n\hat\beta_j^{\text{MCO}}-n\hat\beta_j$. Basta comprobar que {eq}`eq-soft` cumple {eq}`eq-kkt`. Sean $b=\hat\beta_j^{\text{MCO}}$ y $\gamma=\lambda/(2n)$.

- Si $b>\gamma$: {eq}`eq-soft` da $\hat\beta_j=b-\gamma>0$ y $c_j=n(b-\hat\beta_j)=n\gamma=\lambda/2=\frac{\lambda}{2}\operatorname{sgn}(\hat\beta_j)$. ✓
- Si $b<-\gamma$: $\hat\beta_j=b+\gamma<0$ y $c_j=-n\gamma=-\lambda/2=\frac{\lambda}{2}\operatorname{sgn}(\hat\beta_j)$. ✓
- Si $|b|\le\gamma$: $\hat\beta_j=0$ y $|c_j|=n|b|\le n\gamma=\lambda/2$. ✓

Por el {prf:ref}`thm-kkt`, {eq}`eq-soft` es el minimizador.

*Derivación directa (sin adivinar la solución).* $S_\lambda(\boldsymbol{\beta})=\boldsymbol{y}^{T}\boldsymbol{y}-2\boldsymbol{\beta}^{T}X^{T}\boldsymbol{y}+n\boldsymbol{\beta}^{T}\boldsymbol{\beta}+\lambda\|\boldsymbol{\beta}\|_1=\boldsymbol{y}^{T}\boldsymbol{y}+\sum_{j}\big(n\beta_j^{2}-2nb_j\beta_j+\lambda|\beta_j|\big)$ se **separa** por coordenadas. Para $\beta_j>0$, la derivada $2n\beta_j-2nb_j+\lambda$ se anula en $\beta_j=b_j-\gamma$, que es admisible si $b_j>\gamma$; el caso $\beta_j<0$ es simétrico; en otro caso, el mínimo de la parábola por tramos está en el pliegue $\beta_j=0$.
```

```{admonition} Comparación ridge vs. lasso
:class: tip
- **Ridge** contrae **proporcionalmente**: multiplica por $n/(n+\lambda)<1$ y nunca anula un coeficiente no nulo.
- **Lasso** **traslada** hacia cero una cantidad fija $\lambda/(2n)$ y **anula** todo coeficiente con $|\hat\beta^{\text{MCO}}_j|\le\lambda/(2n)$.
- Sesgo: para coeficientes grandes, el de lasso es constante ($\lambda/2n$), mientras que el de ridge crece con $|\beta_j|$ (la fracción $\lambda/(n+\lambda)$ de $\hat\beta^{\text{MCO}}_j$). Compárese con la propiedad (c) de (3.12).
```

## 6.4 Caso general: descenso por coordenadas

Sin ortogonalidad no hay fórmula cerrada, pero el problema **en una sola coordenada** sí la tiene.

```{prf:proposition} Actualización por coordenada
:label: prop-cd

Fijados $\beta_k$, $k\neq j$, sea el residuo parcial $\boldsymbol{r}^{(j)}:=\boldsymbol{y}-\sum_{k\neq j}\boldsymbol{x}_k\beta_k$. El minimizador de $S_\lambda$ en la variable $\beta_j$ es

$$
\beta_j\ \leftarrow\ \mathcal{S}\!\left(\frac{\boldsymbol{x}_j^{T}\boldsymbol{r}^{(j)}}{n},\ \frac{\lambda}{2n}\right).
$$ (eq-cd)

Si $\boldsymbol{\beta}$ es un **punto fijo** de {eq}`eq-cd` para todas las coordenadas, entonces $\boldsymbol{\beta}$ cumple {eq}`eq-kkt` y es un minimizador global.
```

```{prf:proof}
Como función de $\beta_j$, y usando $\boldsymbol{x}_j^{T}\boldsymbol{x}_j=n$,

$$
S_\lambda=\|\boldsymbol{r}^{(j)}-\boldsymbol{x}_j\beta_j\|^{2}+\lambda|\beta_j|+\text{cte}=n\beta_j^{2}-2\beta_j\,\boldsymbol{x}_j^{T}\boldsymbol{r}^{(j)}+\lambda|\beta_j|+\text{cte}.
$$

Es el mismo problema escalar de la demostración del {prf:ref}`thm-soft`, con $b=\boldsymbol{x}_j^{T}\boldsymbol{r}^{(j)}/n$; su solución es {eq}`eq-cd`.

Para el punto fijo: por el {prf:ref}`thm-kkt` aplicado al problema escalar (con $p=1$), $\beta_j$ minimiza en su coordenada si y solo si $\boldsymbol{x}_j^{T}(\boldsymbol{r}^{(j)}-\boldsymbol{x}_j\beta_j)\in\frac{\lambda}{2}\partial|\beta_j|$. Pero $\boldsymbol{r}^{(j)}-\boldsymbol{x}_j\beta_j=\boldsymbol{y}-X\boldsymbol{\beta}$ es el residuo completo. Si esto vale para todo $j$, es exactamente {eq}`eq-kkt`, y por el {prf:ref}`thm-kkt` $\boldsymbol{\beta}$ es un mínimo global.
```

```{prf:algorithm} Lasso por descenso cíclico por coordenadas
:label: alg-cd

**Entrada:** $X$ estandarizada (3.11), $\boldsymbol{y}$ centrada, $\lambda>0$, tolerancia $\tau$.

1. $\boldsymbol{\beta}\leftarrow\boldsymbol{0}$, $\ \boldsymbol{r}\leftarrow\boldsymbol{y}$
2. **Repetir**
    1. **Para** $j=1,\dots,p$:
        1. $\boldsymbol{r}^{(j)}\leftarrow\boldsymbol{r}+\boldsymbol{x}_j\beta_j$
        2. $\beta_j\leftarrow\mathcal{S}\big(\boldsymbol{x}_j^{T}\boldsymbol{r}^{(j)}/n,\ \lambda/(2n)\big)$
        3. $\boldsymbol{r}\leftarrow\boldsymbol{r}^{(j)}-\boldsymbol{x}_j\beta_j$
3. **Hasta** que $\max_j|\Delta\beta_j|<\tau$
4. **Devolver** $\boldsymbol{\beta}$
```

Cada actualización no aumenta $S_\lambda$, porque minimiza en una coordenada. Como la parte no diferenciable $\lambda\|\boldsymbol{\beta}\|_1$ es **separable**, el descenso cíclico converge a un minimizador global {cite}`tseng2001,friedman2007`. Es el método que usa `sklearn.linear_model.Lasso`.

## 6.5 El camino lasso y el algoritmo LARS

### 6.5.1 El camino es lineal a trozos

```{prf:proposition} Linealidad a trozos de $\lambda\mapsto\hat{\boldsymbol{\beta}}(\lambda)$
:label: prop-path

Supongamos que, en un intervalo $\lambda\in(\lambda_1,\lambda_0)$, el conjunto activo $\mathcal{A}=\{j:\hat\beta_j(\lambda)\neq0\}$ y los signos $\boldsymbol{s}_{\mathcal{A}}=\operatorname{sgn}(\hat{\boldsymbol{\beta}}_{\mathcal{A}})$ son constantes, y que $X_{\mathcal{A}}$ (las columnas activas) tiene rango completo. Entonces, en ese intervalo,

$$
\hat{\boldsymbol{\beta}}_{\mathcal{A}}(\lambda)=(X_{\mathcal{A}}^{T}X_{\mathcal{A}})^{-1}\Big(X_{\mathcal{A}}^{T}\boldsymbol{y}-\frac{\lambda}{2}\boldsymbol{s}_{\mathcal{A}}\Big),\qquad \hat{\boldsymbol{\beta}}_{\mathcal{A}^c}(\lambda)=\boldsymbol{0},
$$ (eq-path)

que es **afín** en $\lambda$, con dirección $\ \dfrac{d\hat{\boldsymbol{\beta}}_{\mathcal{A}}}{d(-\lambda)}=\boldsymbol{d}_{\mathcal{A}}:=\tfrac12(X_{\mathcal{A}}^{T}X_{\mathcal{A}})^{-1}\boldsymbol{s}_{\mathcal{A}}$.
```

```{prf:proof}
Por {eq}`eq-kkt`, las coordenadas activas cumplen $X_{\mathcal{A}}^{T}(\boldsymbol{y}-X_{\mathcal{A}}\hat{\boldsymbol{\beta}}_{\mathcal{A}})=\frac{\lambda}{2}\boldsymbol{s}_{\mathcal{A}}$, porque las inactivas no contribuyen al ajuste. Despejando, $X_{\mathcal{A}}^{T}X_{\mathcal{A}}\hat{\boldsymbol{\beta}}_{\mathcal{A}}=X_{\mathcal{A}}^{T}\boldsymbol{y}-\frac{\lambda}{2}\boldsymbol{s}_{\mathcal{A}}$, y como la matriz es invertible se obtiene {eq}`eq-path`. Derivando respecto de $-\lambda$ se obtiene $\boldsymbol{d}_{\mathcal{A}}$.
```

El camino solo cambia de dirección en $\lambda$ donde cambia $\mathcal{A}$. Hay dos tipos de **eventos** al disminuir $\lambda$ en $t>0$ (escribimos $\boldsymbol{a}:=X^{T}X_{\mathcal{A}}\boldsymbol{d}_{\mathcal{A}}$, de modo que $\boldsymbol{c}(\lambda-t)=\boldsymbol{c}(\lambda)-t\boldsymbol{a}$):

- **Entrada** de $j\notin\mathcal{A}$: su correlación alcanza la de las activas, $c_j-ta_j=\pm\frac{\lambda-t}{2}$, es decir,

$$
t_j^{\pm}=\frac{c_j\mp\lambda/2}{a_j\mp1/2}\quad(\text{se toma el menor valor positivo}).
$$ (eq-entrada)

  Las activas cumplen $\boldsymbol{c}_{\mathcal{A}}(\lambda-t)=\frac{\lambda}{2}\boldsymbol{s}_{\mathcal{A}}-tX_{\mathcal{A}}^{T}X_{\mathcal{A}}\boldsymbol{d}_{\mathcal{A}}=\frac{\lambda-t}{2}\boldsymbol{s}_{\mathcal{A}}$: permanecen **empatadas**.
- **Salida** de $j\in\mathcal{A}$: el coeficiente cruza cero, $\hat\beta_j+td_j=0$, es decir, $t_j=-\hat\beta_j/d_j$ (si es positivo). Más allá de ese punto, $\operatorname{sgn}(\hat\beta_j)\neq s_j$ violaría {eq}`eq-kkt`, de modo que $j$ debe **salir** de $\mathcal{A}$.

### 6.5.2 LARS de Efron et al. (2004)

**LARS** {cite}`efron2004` construye el ajuste $\hat{\boldsymbol{\mu}}=X\hat{\boldsymbol{\beta}}$ por pasos, con columnas de norma unitaria (aquí, $\boldsymbol{x}_j/\sqrt n$):

```{prf:algorithm} LARS (Least Angle Regression) con modificación lasso
:label: alg-lars

1. $\hat{\boldsymbol{\mu}}\leftarrow\boldsymbol{0}$. Calcular $\hat{\boldsymbol{c}}=X^{T}(\boldsymbol{y}-\hat{\boldsymbol{\mu}})$, $\ \hat C=\max_j|\hat c_j|$ y $\mathcal{A}=\{j:|\hat c_j|=\hat C\}$, con signos $s_j=\operatorname{sgn}(\hat c_j)$.
2. Con las columnas firmadas $X_{\mathcal{A}}=(s_j\boldsymbol{x}_j)_{j\in\mathcal{A}}$, formar $\mathcal{G}_{\mathcal{A}}=X_{\mathcal{A}}^{T}X_{\mathcal{A}}$, $\ A_{\mathcal{A}}=(\boldsymbol{1}^{T}\mathcal{G}_{\mathcal{A}}^{-1}\boldsymbol{1})^{-1/2}$, $\ \boldsymbol{w}_{\mathcal{A}}=A_{\mathcal{A}}\mathcal{G}_{\mathcal{A}}^{-1}\boldsymbol{1}$ y el **vector equiangular** $\boldsymbol{u}_{\mathcal{A}}=X_{\mathcal{A}}\boldsymbol{w}_{\mathcal{A}}$.
3. Con $\boldsymbol{a}=X^{T}\boldsymbol{u}_{\mathcal{A}}$, el paso es

   $$\hat\gamma=\min_{j\notin\mathcal{A}}{}^{+}\left\{\frac{\hat C-\hat c_j}{A_{\mathcal{A}}-a_j},\ \frac{\hat C+\hat c_j}{A_{\mathcal{A}}+a_j}\right\}.$$
4. **Modificación lasso:** con $d_j=s_jw_{\mathcal{A},j}$ y $\gamma_j=-\hat\beta_j/d_j$, sea $\tilde\gamma=\min_{\gamma_j>0}\gamma_j$. Si $\tilde\gamma<\hat\gamma$, avanzar solo $\tilde\gamma$ y **retirar** de $\mathcal{A}$ la variable que se anula.
5. $\hat{\boldsymbol{\mu}}\leftarrow\hat{\boldsymbol{\mu}}+\gamma\boldsymbol{u}_{\mathcal{A}}$; actualizar $\mathcal{A}$ y volver a 2 hasta que $\hat C=0$ (o se agoten las variables).
```

```{prf:lemma} Propiedad equiangular
:label: lem-equiangular

$\ X_{\mathcal{A}}^{T}\boldsymbol{u}_{\mathcal{A}}=A_{\mathcal{A}}\boldsymbol{1}_{\mathcal{A}}\ $ y $\ \|\boldsymbol{u}_{\mathcal{A}}\|=1$. Es decir, $\boldsymbol{u}_{\mathcal{A}}$ forma el **mismo ángulo** con todas las columnas activas (firmadas). Además, $\boldsymbol{u}_{\mathcal{A}}$ es un múltiplo positivo de $X_{\mathcal{A}}\boldsymbol{d}_{\mathcal{A}}$ de la {prf:ref}`prop-path`.
```

```{prf:proof}
$X_{\mathcal{A}}^{T}\boldsymbol{u}_{\mathcal{A}}=\mathcal{G}_{\mathcal{A}}\boldsymbol{w}_{\mathcal{A}}=A_{\mathcal{A}}\mathcal{G}_{\mathcal{A}}\mathcal{G}_{\mathcal{A}}^{-1}\boldsymbol{1}=A_{\mathcal{A}}\boldsymbol{1}$. Además, $\|\boldsymbol{u}_{\mathcal{A}}\|^{2}=\boldsymbol{w}_{\mathcal{A}}^{T}\mathcal{G}_{\mathcal{A}}\boldsymbol{w}_{\mathcal{A}}=A_{\mathcal{A}}^{2}\,\boldsymbol{1}^{T}\mathcal{G}_{\mathcal{A}}^{-1}\boldsymbol{1}=1$ por la definición de $A_{\mathcal{A}}$.

Para la última afirmación: si $D_s=\operatorname{diag}(\boldsymbol{s}_{\mathcal{A}})$ y $\bar X_{\mathcal{A}}$ son las columnas **sin** firmar, entonces $X_{\mathcal{A}}=\bar X_{\mathcal{A}}D_s$ y $\mathcal{G}_{\mathcal{A}}^{-1}=D_s(\bar X_{\mathcal{A}}^{T}\bar X_{\mathcal{A}})^{-1}D_s$ (porque $D_s^{-1}=D_s$). Así, $\boldsymbol{u}_{\mathcal{A}}=A_{\mathcal{A}}\bar X_{\mathcal{A}}D_sD_s(\bar X_{\mathcal{A}}^{T}\bar X_{\mathcal{A}})^{-1}D_s\boldsymbol{1}=A_{\mathcal{A}}\bar X_{\mathcal{A}}(\bar X_{\mathcal{A}}^{T}\bar X_{\mathcal{A}})^{-1}\boldsymbol{s}_{\mathcal{A}}=2A_{\mathcal{A}}\,\bar X_{\mathcal{A}}\boldsymbol{d}_{\mathcal{A}}$, con $2A_{\mathcal{A}}>0$.
```

```{prf:proof}
**Por qué el paso $\hat\gamma$ de LARS coincide con los eventos de entrada.** Al avanzar $\hat{\boldsymbol{\mu}}+\gamma\boldsymbol{u}_{\mathcal{A}}$, las correlaciones pasan a ser $c_j(\gamma)=\hat c_j-\gamma a_j$. Para las activas, por el {prf:ref}`lem-equiangular`, $|c_j(\gamma)|=\hat C-\gamma A_{\mathcal{A}}$: todas bajan **al mismo ritmo**, que es la condición $|c_j|=\lambda/2$ de {eq}`eq-kkt` con $\lambda/2=\hat C-\gamma A_{\mathcal{A}}$. Una inactiva $j$ las alcanza cuando $\hat c_j-\gamma a_j=\pm(\hat C-\gamma A_{\mathcal{A}})$, es decir, en $\gamma=(\hat C-\hat c_j)/(A_{\mathcal{A}}-a_j)$ o $\gamma=(\hat C+\hat c_j)/(A_{\mathcal{A}}+a_j)$; el primer valor positivo es $\hat\gamma$. La modificación lasso es la "salida" de la Sección 6.5.1. Así, LARS modificado recorre exactamente los eventos del camino lasso, y {cite}`efron2004` (Teorema 1) prueba que produce **todas** las soluciones lasso $\hat{\boldsymbol{\beta}}(\lambda)$, $\lambda\in[0,\lambda_{\max}]$, bajo la condición de que las variables entran o salen de una en una. El costo es el de un ajuste por mínimos cuadrados.
```

## 6.6 Verificación numérica

Se usa el conjunto `diabetes` ($n=442$, $p=10$), estandarizado según (3.11). Para comparar con `scikit-learn`, nótese que `Lasso` y `lars_path` minimizan $\frac{1}{2n}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^{2}+\alpha\|\boldsymbol{\beta}\|_1$. Multiplicando por $2n$ se obtiene {eq}`eq-lasso` con $\lambda=2n\alpha$.

```{code-cell} ipython3
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, lars_path

X, y = load_diabetes(return_X_y=True)
n, p = X.shape
y = y - y.mean()                           # sum y_i = 0
X = (X - X.mean(0)) / X.std(0)             # sum x_ij = 0, sum x_ij^2 = n   (3.11)

soft = lambda z, g: np.sign(z)*np.maximum(np.abs(z) - g, 0.0)    # S(z, gamma)

def lasso_cd(X, y, lam, tol=1e-12, max_iter=10_000):
    """Algoritmo de descenso por coordenadas (umbralización suave)."""
    n, p = X.shape; b = np.zeros(p); r = y.copy()
    for it in range(max_iter):
        delta = 0.0
        for j in range(p):
            rj = r + X[:, j]*b[j]
            nuevo = soft(X[:, j] @ rj / n, lam/(2*n))
            delta = max(delta, abs(nuevo - b[j])); b[j] = nuevo
            r = rj - X[:, j]*b[j]
        if delta < tol: break
    return b, it + 1

lam_max = 2*np.max(np.abs(X.T @ y))
print(f"lambda_max = 2 max|x_j^T y| = {lam_max:.1f}")
for lam in [0.5*lam_max, 0.1*lam_max, 0.01*lam_max]:
    b, it = lasso_cd(X, y, lam)
    sk = Lasso(alpha=lam/(2*n), fit_intercept=False, tol=1e-14, max_iter=100_000).fit(X, y).coef_
    c = X.T @ (y - X @ b)
    act = b != 0
    kkt_act = np.abs(c[act] - lam/2*np.sign(b[act])).max() if act.any() else 0
    kkt_ina = np.abs(c[~act]).max() - lam/2 if (~act).any() else -np.inf
    print(f"lambda={lam:9.1f}: {it:4d} iter | activos={act.sum():2d} | "
          f"máx|CD-sklearn|={np.abs(b-sk).max():.1e} | KKT activos {kkt_act:.1e} | máx|c_j| inactivos - lambda/2 = {kkt_ina:.2f} (<=0)")
```

```{code-cell} ipython3
def lasso_homotopia(X, y, tol=1e-9):
    """Camino lasso completo (LARS con modificación lasso), parametrizado por lambda."""
    n, p = X.shape
    b = np.zeros(p); c = X.T @ y
    lam = 2*np.max(np.abs(c)); A = [int(np.argmax(np.abs(c)))]
    lams, B = [lam], [b.copy()]
    while lam > 0:
        s = np.sign(c[A]); XA = X[:, A]
        d = np.linalg.solve(XA.T @ XA, s) / 2          # dirección d_A (Prop. del camino)
        a = X.T @ (XA @ d)
        eventos = [(lam, "fin", None)]
        for j in set(range(p)) - set(A):               # entradas, ecuación (eq-entrada)
            for sg in (1, -1):
                if abs(a[j] - sg/2) > 1e-15:
                    t = (c[j] - sg*lam/2) / (a[j] - sg/2)
                    if t > tol*lams[0]: eventos.append((t, "entra", j))
        for k, j in enumerate(A):                      # salidas (cruce por cero)
            if abs(d[k]) > 1e-15 and -b[j]/d[k] > 1e-12:
                eventos.append((-b[j]/d[k], "sale", j))
        t, tipo, j = min(eventos, key=lambda e: e[0])
        b[A] += t*d; lam -= t; c = X.T @ (y - X @ b)
        if tipo == "sale": b[j] = 0.0; A.remove(j)
        elif tipo == "entra": A.append(j)
        lams.append(max(lam, 0.0)); B.append(b.copy())
        if tipo == "fin": break
    return np.array(lams), np.array(B)

lams, B = lasso_homotopia(X, y)
alphas, _, coefs = lars_path(X, y, method="lasso")
print("número de quiebres:", len(lams), "(propio) vs", len(alphas), "(sklearn lars_path)")
print("máx |lambda/(2n) - alpha_sklearn| =", np.abs(lams/(2*n) - alphas).max())
print("máx |beta propio - beta sklearn| en los quiebres =", np.abs(B.T - coefs).max())
b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
print("en lambda = 0 el camino termina en MCO:", np.allclose(B[-1], b_ols))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

# (a) Diseño ortogonal: lasso (umbralización suave) vs ridge (contracción proporcional)
bmco = np.linspace(-3, 3, 601); n0, lam0 = 50, 100
ax[0].plot(bmco, bmco, ":", c="gray", label="MCO")
ax[0].plot(bmco, n0/(n0+lam0)*bmco, label=r"ridge $\frac{n}{n+\lambda}\hat\beta^{MCO}$")
ax[0].plot(bmco, soft(bmco, lam0/(2*n0)), lw=2, label=r"lasso $\mathcal{S}(\hat\beta^{MCO},\lambda/2n)$")
ax[0].axvspan(-lam0/(2*n0), lam0/(2*n0), color="0.9", label="zona muerta: $\\hat\\beta^L=0$")
ax[0].set_xlabel(r"$\hat\beta_j^{MCO}$"); ax[0].set_title("Diseño ortogonal ($X^TX=nI$)"); ax[0].legend(fontsize=8)

# (b) Camino lasso del ejemplo diabetes: lineal a trozos en lambda
nombres = load_diabetes().feature_names
for j in range(p):
    ax[1].plot(lams, B[:, j], "-o", ms=3, label=nombres[j])
ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel(r"$\hat\beta_j(\lambda)$"); ax[1].invert_xaxis()
ax[1].set_title("Camino lasso (LARS): lineal a trozos"); ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
# Comprobación de la fórmula cerrada en un diseño ortogonal estandarizado
rng = np.random.default_rng(3)
n, p = 60, 6
G = rng.normal(size=(n, p)); G -= G.mean(0)            # columnas centradas
Q, _ = np.linalg.qr(G); Xo = np.sqrt(n)*Q                # X^T X = n I, sum x_ij = 0, sum x_ij^2 = n
yo = Xo @ np.array([3, -2, 0.5, 0, 0, 0.1]) + rng.normal(size=n); yo -= yo.mean()
lam = 40.0
b_mco = Xo.T @ yo / n
print("¿X^T X = n I?", np.allclose(Xo.T @ Xo, n*np.eye(p)), "| ¿columnas centradas?", np.allclose(Xo.sum(0), 0))
print("fórmula soft  :", soft(b_mco, lam/(2*n)).round(5))
print("desc. coord.  :", lasso_cd(Xo, yo, lam)[0].round(5))
print("ridge n/(n+λ) :", (n/(n+lam)*b_mco).round(5))
```

Las tres comprobaciones confirman la teoría: el descenso por coordenadas coincide con `sklearn` y satisface las condiciones KKT; la homotopía/LARS reproduce exactamente los quiebres de `lars_path` y termina en mínimos cuadrados; y en el diseño ortogonal, lasso es la umbralización suave, que anula los coeficientes pequeños, mientras que ridge solo los escala.
