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

# 1. K-NN: error verdadero y error empírico (Definición 2.3)

## Enunciado

```{figure} images/enunciado_def23.png
:width: 90%
:name: fig-enun-def23

Enunciado: Definición 2.3 (Error empírico y verdadero revisado). Se pide **demostrar** lo que la definición afirma.
```

La Definición 2.3 encadena tres expresiones del **riesgo** de una regla $h:\mathcal{X}\to\mathcal{Y}$,

$$
L_{\mathcal{D}}(h):=\underset{(\boldsymbol{x},y)\sim\mathcal{D}}{\mathbb{P}}[h(\boldsymbol{x})\neq y]:=\mathcal{D}(\{(\boldsymbol{x},y):~h(\boldsymbol{x})\neq y\}),
$$ (eq-22)

y presenta el **riesgo empírico** $L_{S}(h)=|\{i\in[m]: h(\boldsymbol{x}_i)\neq y_i\}|/m$ como la cantidad calculable que sustituye a $L_{\mathcal{D}}(h)$, con el objetivo de encontrar $h$ que (aproximadamente) minimice el riesgo verdadero. Justo después, las notas afirman que el **predictor óptimo de Bayes** $f_{\mathcal{D}}$ cumple $L_{\mathcal{D}}(f_{\mathcal{D}})\le L_{\mathcal{D}}(g)$ para todo clasificador $g$ ("se puede verificar que...").

Lo que hay que justificar se organiza en cuatro proposiciones:

1. Las igualdades de {eq}`eq-22` y su equivalencia con la pérdida esperada $\mathbb{E}[\ell_{0\text{-}1}]$ de la Ecuación (2.4).
2. Una fórmula de $L_{\mathcal{D}}(h)$ en términos de la probabilidad condicional $\eta(\boldsymbol{x})=\mathbb{P}[y=1\mid\boldsymbol{x}]$.
3. Que $L_{S}(h)$ es un estimador insesgado y consistente de $L_{\mathcal{D}}(h)$, lo que justifica usar el riesgo empírico en lugar del verdadero.
4. La optimalidad del predictor de Bayes, junto con la identidad $L_{\mathcal{D}}(h^{\star})=\mathbb{E}_{\boldsymbol{x}}[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}]$, que se usa en el Lema 2.1 (Capítulo 2).

**Marco común.** $\mathcal{D}$ es una medida de probabilidad sobre $\mathcal{X}\times\mathcal{Y}$ con $\mathcal{Y}=\{0,1\}$, y $h$ es medible, de modo que el conjunto

$$
A_h:=\{(\boldsymbol{x},y)\in\mathcal{X}\times\mathcal{Y}:~h(\boldsymbol{x})\neq y\}
$$

es medible: es la preimagen del conjunto $\{(a,b): a\neq b\}$ por la función medible $(\boldsymbol{x},y)\mapsto(h(\boldsymbol{x}),y)$. $\mathcal{D}_{x}$ denota la marginal de $\boldsymbol{x}$.

## 1.1 Las tres expresiones del riesgo coinciden

```{prf:proposition} Equivalencia de las definiciones del riesgo
:label: prop-riesgo-equiv

Para toda $h$ medible,

$$
\underset{(\boldsymbol{x},y)\sim\mathcal{D}}{\mathbb{P}}[h(\boldsymbol{x})\neq y]
=\mathcal{D}(A_h)
=\underset{(\boldsymbol{x},y)\sim\mathcal{D}}{\mathbb{E}}\big[\mathbb{1}_{[h(\boldsymbol{x})\neq y]}\big]
=\underset{z\sim\mathcal{D}}{\mathbb{E}}\big[\ell_{0\text{-}1}(h,z)\big].
$$

En consecuencia, las definiciones (2.2) y (2.4) del riesgo, esta última con la pérdida 0-1, son la misma.
```

```{prf:proof}
Sea $(\boldsymbol{x},y)$ un par aleatorio con ley $\mathcal{D}$, es decir, $\mathbb{P}[(\boldsymbol{x},y)\in B]=\mathcal{D}(B)$ para todo $B$ medible.

*Primera igualdad.* El suceso $\{h(\boldsymbol{x})\neq y\}$ es exactamente $\{(\boldsymbol{x},y)\in A_h\}$. Por definición de ley,

$$
\mathbb{P}[h(\boldsymbol{x})\neq y]=\mathbb{P}[(\boldsymbol{x},y)\in A_h]=\mathcal{D}(A_h).
$$

*Segunda igualdad.* La indicadora $\mathbb{1}_{[h(\boldsymbol{x})\neq y]}=\mathbb{1}_{A_h}(\boldsymbol{x},y)$ es una función simple con valores en $\{0,1\}$, y por definición de la integral de Lebesgue,

$$
\mathbb{E}\big[\mathbb{1}_{A_h}(\boldsymbol{x},y)\big]=\int_{\mathcal{X}\times\mathcal{Y}}\mathbb{1}_{A_h}\,d\mathcal{D}=1\cdot\mathcal{D}(A_h)+0\cdot\mathcal{D}(A_h^{c})=\mathcal{D}(A_h).
$$

Este es el hecho que las notas enuncian como $\mathbb{E}_{\alpha\sim\mathcal{D}}[\alpha]=\mathbb{P}_{\alpha\sim\mathcal{D}}[\alpha=1]$ para una variable $\alpha\in\{0,1\}$; aquí se aplica con $\alpha=\mathbb{1}_{[h(\boldsymbol{x})\neq y]}$.

*Tercera igualdad.* Por definición, $\ell_{0\text{-}1}(h,(\boldsymbol{x},y))=0$ si $h(\boldsymbol{x})=y$ y $1$ si $h(\boldsymbol{x})\neq y$; es decir, $\ell_{0\text{-}1}(h,(\boldsymbol{x},y))=\mathbb{1}_{[h(\boldsymbol{x})\neq y]}$. Tomando esperanzas se obtiene la igualdad.
```

## 1.2 El riesgo en términos de $\eta(\boldsymbol{x})$

```{prf:proposition} Descomposición condicional del riesgo
:label: prop-riesgo-eta

Sea $\eta(\boldsymbol{x})=\mathbb{P}[y=1\mid\boldsymbol{x}]$. Para toda $h:\mathcal{X}\to\{0,1\}$,

$$
L_{\mathcal{D}}(h)=\underset{\boldsymbol{x}\sim\mathcal{D}_{x}}{\mathbb{E}}\Big[\eta(\boldsymbol{x})\,\mathbb{1}_{[h(\boldsymbol{x})=0]}+\big(1-\eta(\boldsymbol{x})\big)\,\mathbb{1}_{[h(\boldsymbol{x})=1]}\Big].
$$ (eq-riesgo-eta)

Es decir, $L_{\mathcal{D}}(h)=\mathbb{E}_{\boldsymbol{x}}[c_h(\boldsymbol{x})]$, donde $c_h(\boldsymbol{x})$ es el **error condicional** en $\boldsymbol{x}$: vale $\eta(\boldsymbol{x})$ si $h$ predice $0$ y $1-\eta(\boldsymbol{x})$ si predice $1$.
```

```{prf:proof}
Como $h(\boldsymbol{x})$ y $y$ solo toman los valores $0$ y $1$, el error ocurre exactamente en dos casos disjuntos: ($h(\boldsymbol{x})=0$ y $y=1$) o ($h(\boldsymbol{x})=1$ y $y=0$). Por lo tanto, puntualmente,

$$
\mathbb{1}_{[h(\boldsymbol{x})\neq y]}=\mathbb{1}_{[h(\boldsymbol{x})=0]}\mathbb{1}_{[y=1]}+\mathbb{1}_{[h(\boldsymbol{x})=1]}\mathbb{1}_{[y=0]}.
$$

Por la {prf:ref}`prop-riesgo-equiv` y la propiedad de la torre, $\mathbb{E}[\,\cdot\,]=\mathbb{E}_{\boldsymbol{x}}\big[\mathbb{E}[\,\cdot\mid\boldsymbol{x}]\big]$:

$$
L_{\mathcal{D}}(h)=\mathbb{E}\big[\mathbb{1}_{[h(\boldsymbol{x})\neq y]}\big]
=\underset{\boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\Big[\mathbb{E}\big[\mathbb{1}_{[h(\boldsymbol{x})=0]}\mathbb{1}_{[y=1]}+\mathbb{1}_{[h(\boldsymbol{x})=1]}\mathbb{1}_{[y=0]}\,\big|\,\boldsymbol{x}\big]\Big].
$$

Dado $\boldsymbol{x}$, los factores $\mathbb{1}_{[h(\boldsymbol{x})=0]}$ y $\mathbb{1}_{[h(\boldsymbol{x})=1]}$ son constantes ($h$ es determinista), así que salen de la esperanza condicional. Además, $\mathbb{E}[\mathbb{1}_{[y=1]}\mid\boldsymbol{x}]=\mathbb{P}[y=1\mid\boldsymbol{x}]=\eta(\boldsymbol{x})$ y $\mathbb{E}[\mathbb{1}_{[y=0]}\mid\boldsymbol{x}]=1-\eta(\boldsymbol{x})$. Sustituyendo se obtiene {eq}`eq-riesgo-eta`.
```

## 1.3 El riesgo empírico estima el riesgo verdadero

La definición introduce $L_S(h)$ porque $\mathcal{D}$ es desconocida. El siguiente resultado justifica esa sustitución para una $h$ **fija** (elegida sin mirar $S$).

```{prf:proposition} Insesgamiento y consistencia de $L_S(h)$
:label: prop-insesgado

Sea $S=((\boldsymbol{x}_1,y_1),\dots,(\boldsymbol{x}_m,y_m))\sim\mathcal{D}^{m}$ (i.i.d.) y $h$ fija. Entonces

1. $\displaystyle\underset{S\sim\mathcal{D}^m}{\mathbb{E}}[L_S(h)]=L_{\mathcal{D}}(h)$,
2. $\displaystyle\operatorname{Var}[L_S(h)]=\frac{L_{\mathcal{D}}(h)\,(1-L_{\mathcal{D}}(h))}{m}\le\frac{1}{4m}$,
3. $L_S(h)\xrightarrow[m\to\infty]{}L_{\mathcal{D}}(h)$ casi seguramente, y además, para todo $\epsilon>0$, $\ \mathbb{P}\big[|L_S(h)-L_{\mathcal{D}}(h)|>\epsilon\big]\le 2e^{-2m\epsilon^{2}}$ (Hoeffding).
```

```{prf:proof}
Escribimos el conteo del numerador como una suma de indicadoras:

$$
L_S(h)=\frac{|\{i\in[m]: h(\boldsymbol{x}_i)\neq y_i\}|}{m}=\frac{1}{m}\sum_{i=1}^{m}B_i,\qquad B_i:=\mathbb{1}_{[h(\boldsymbol{x}_i)\neq y_i]}.
$$

Como los pares $(\boldsymbol{x}_i,y_i)$ son i.i.d. con ley $\mathcal{D}$ y $B_i$ es la misma función medible aplicada a cada par, las $B_i$ son i.i.d. Por la {prf:ref}`prop-riesgo-equiv`, $\mathbb{P}[B_i=1]=L_{\mathcal{D}}(h)$, es decir, $B_i\sim\text{Bernoulli}(L_{\mathcal{D}}(h))$.

1. Por linealidad de la esperanza, $\mathbb{E}[L_S(h)]=\frac1m\sum_{i}\mathbb{E}[B_i]=\frac1m\cdot m\,L_{\mathcal{D}}(h)=L_{\mathcal{D}}(h)$.
2. Por independencia, las varianzas se suman: $\operatorname{Var}[L_S(h)]=\frac{1}{m^2}\sum_i\operatorname{Var}[B_i]=\frac{1}{m^2}\,m\,L_{\mathcal{D}}(h)(1-L_{\mathcal{D}}(h))$. Como $t(1-t)\le 1/4$ en $[0,1]$ (su máximo está en $t=1/2$), se obtiene la cota.
3. La convergencia casi segura es la ley fuerte de los grandes números aplicada a las $B_i$, que son integrables. La cota exponencial es la desigualdad de Hoeffding para variables i.i.d. con valores en $[0,1]$.
```

```{admonition} Observación
:class: note
Si $h$ se elige **usando** $S$ (por ejemplo, $h=\text{ERM}_{\mathcal{H}}(S)$), las $B_i$ dejan de ser independientes de $h$ y la {prf:ref}`prop-insesgado` no se aplica directamente. Por eso $L_S(h_S)$ suele ser optimista y hace falta la teoría de generalización (clases finitas, dimensión VC o, para 1-NN, el Lema 2.1).
```

## 1.4 Optimalidad del predictor de Bayes

Las notas definen $f_{\mathcal{D}}(\boldsymbol{x})=1$ si $\mathbb{P}[y=1\mid\boldsymbol{x}]\ge 1/2$ y $0$ en otro caso (Ecuación 2.3), y en la sección de 1-NN usan $h^{\star}(\boldsymbol{x})=\mathbb{1}_{[\eta(\boldsymbol{x})>1/2]}$.

```{prf:theorem} El predictor de Bayes es óptimo
:label: thm-bayes

Para todo clasificador medible $g:\mathcal{X}\to\{0,1\}$,

$$
L_{\mathcal{D}}(g)-L_{\mathcal{D}}(f_{\mathcal{D}})=\underset{\boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\Big[\,|2\eta(\boldsymbol{x})-1|\;\mathbb{1}_{[g(\boldsymbol{x})\neq f_{\mathcal{D}}(\boldsymbol{x})]}\Big]\;\ge\;0 .
$$ (eq-exceso-bayes)

En particular, $L_{\mathcal{D}}(f_{\mathcal{D}})\le L_{\mathcal{D}}(g)$. Además,

$$
L_{\mathcal{D}}(f_{\mathcal{D}})=L_{\mathcal{D}}(h^{\star})=\underset{\boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\big[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}\big].
$$ (eq-riesgo-bayes)
```

```{prf:proof}
Por la {prf:ref}`prop-riesgo-eta`, $L_{\mathcal{D}}(g)=\mathbb{E}_{\boldsymbol{x}}[c_g(\boldsymbol{x})]$, con $c_g(\boldsymbol{x})=\eta(\boldsymbol{x})$ si $g(\boldsymbol{x})=0$ y $c_g(\boldsymbol{x})=1-\eta(\boldsymbol{x})$ si $g(\boldsymbol{x})=1$. Basta entonces comparar los errores condicionales **punto a punto**.

*Paso 1: error condicional de $f_{\mathcal{D}}$.* Fijemos $\boldsymbol{x}$.

- Si $\eta(\boldsymbol{x})\ge1/2$: $f_{\mathcal{D}}(\boldsymbol{x})=1$ y $c_{f}(\boldsymbol{x})=1-\eta(\boldsymbol{x})$. Como $\eta(\boldsymbol{x})\ge 1/2\ge1-\eta(\boldsymbol{x})$, este valor es $\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}$.
- Si $\eta(\boldsymbol{x})<1/2$: $f_{\mathcal{D}}(\boldsymbol{x})=0$ y $c_{f}(\boldsymbol{x})=\eta(\boldsymbol{x})$, que también es el mínimo porque $\eta(\boldsymbol{x})<1/2<1-\eta(\boldsymbol{x})$.

Luego $c_f(\boldsymbol{x})=\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}$ para todo $\boldsymbol{x}$.

*Paso 2: comparación.* Si $g(\boldsymbol{x})=f_{\mathcal{D}}(\boldsymbol{x})$, entonces $c_g(\boldsymbol{x})=c_f(\boldsymbol{x})$. Si $g(\boldsymbol{x})\neq f_{\mathcal{D}}(\boldsymbol{x})$, $g$ toma la otra etiqueta y su error condicional es el *otro* número del par $\{\eta,1-\eta\}$, es decir, el máximo. Así,

$$
\begin{aligned}
c_g(\boldsymbol{x})-c_f(\boldsymbol{x})&=\big(\max\{\eta,1-\eta\}-\min\{\eta,1-\eta\}\big)\mathbb{1}_{[g\neq f_{\mathcal{D}}]}\\
&=|\eta(\boldsymbol{x})-(1-\eta(\boldsymbol{x}))|\,\mathbb{1}_{[g\neq f_{\mathcal{D}}]}=|2\eta(\boldsymbol{x})-1|\,\mathbb{1}_{[g\neq f_{\mathcal{D}}]},
\end{aligned}
$$

donde se usó $\max\{a,b\}-\min\{a,b\}=|a-b|$. Tomando $\mathbb{E}_{\boldsymbol{x}}$ se obtiene {eq}`eq-exceso-bayes`, que es $\ge0$ por ser la esperanza de una variable no negativa.

*Paso 3: $h^{\star}$ tiene el mismo riesgo.* $h^{\star}$ y $f_{\mathcal{D}}$ solo difieren donde $\eta(\boldsymbol{x})=1/2$, y ahí $|2\eta(\boldsymbol{x})-1|=0$. Aplicando {eq}`eq-exceso-bayes` con $g=h^{\star}$ resulta $L_{\mathcal{D}}(h^{\star})=L_{\mathcal{D}}(f_{\mathcal{D}})$. Por el Paso 1, $L_{\mathcal{D}}(f_{\mathcal{D}})=\mathbb{E}_{\boldsymbol{x}}[c_f(\boldsymbol{x})]=\mathbb{E}_{\boldsymbol{x}}[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}]$, que es {eq}`eq-riesgo-bayes`.
```

```{admonition} Interpretación
:class: tip
- {eq}`eq-exceso-bayes` dice que una regla pierde frente a Bayes **solo** donde discrepa de ella, y que la pérdida en esos puntos es mayor cuanto más lejos está $\eta(\boldsymbol{x})$ de $1/2$, es decir, cuanto más "obvia" era la etiqueta.
- {eq}`eq-riesgo-bayes` es el error irreducible: es nulo solo en el caso determinista $\eta(\boldsymbol{x})\in\{0,1\}$, como señala el ejemplo de las notas ($\eta=0.73$ todavía deja un error condicional de $0.27$).
```

## 1.5 Verificación numérica

Se construye una distribución con $\mathcal{X}=[0,1]$, $\boldsymbol{x}\sim U[0,1]$ y $\eta(x)=0.5+0.4\sin(2\pi x)$. Se comparan:

- $L_{\mathcal{D}}(h)$ calculado por integración de {eq}`eq-riesgo-eta`;
- $\mathcal{D}(A_h)$ estimado por Monte Carlo;
- la media de $L_S(h)$ sobre muchas muestras con $m=50$ ({prf:ref}`prop-insesgado`);
- el riesgo de varias reglas frente al de Bayes ({prf:ref}`thm-bayes`).

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng(0)

eta = lambda x: 0.5 + 0.4*np.sin(2*np.pi*x)
reglas = {
    "Bayes h*(x)=1[eta>1/2]": lambda x: (eta(x) > 0.5).astype(int),
    "umbral x<0.3":           lambda x: (x < 0.3).astype(int),
    "constante 1":            lambda x: np.ones_like(x, dtype=int),
    "aleatoria fija":         lambda x: (np.sin(37*x) > 0).astype(int),
}

grid = np.linspace(0, 1, 2_000_001)            # integración numérica en [0,1]
def riesgo_integral(h):
    hx = h(grid)
    return np.mean(eta(grid)*(hx == 0) + (1 - eta(grid))*(hx == 1))

def muestra(m):
    x = rng.uniform(size=m)
    y = (rng.uniform(size=m) < eta(x)).astype(int)
    return x, y

xb, yb = muestra(2_000_000)                    # Monte Carlo de D(A_h)
print(f"{'regla':26s} {'L_D (integral)':>15s} {'D(A_h) (MC)':>12s} {'E[L_S], m=50':>13s} {'Var L_S':>9s} {'L(1-L)/m':>9s}")
for nombre, h in reglas.items():
    LD = riesgo_integral(h)
    MC = np.mean(h(xb) != yb)
    LS = np.array([np.mean(h(x) != y) for x, y in (muestra(50) for _ in range(20000))])
    print(f"{nombre:26s} {LD:15.4f} {MC:12.4f} {LS.mean():13.4f} {LS.var():9.5f} {LD*(1-LD)/50:9.5f}")

print("\nE[min(eta,1-eta)] =", round(np.mean(np.minimum(eta(grid), 1 - eta(grid))), 4))
```

Las columnas coinciden con la teoría: las tres formas del riesgo dan el mismo número, $\mathbb{E}[L_S]=L_{\mathcal{D}}$ y la varianza empírica es $L(1-L)/m$. La regla de Bayes tiene el menor riesgo, que es igual a $\mathbb{E}[\min\{\eta,1-\eta\}]$.
