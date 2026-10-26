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

# 2. K-NN: el caso $\min\{\eta,1-\eta\}=1-\eta(\boldsymbol{x})$ en el Lema 2.1

## Enunciado

```{figure} images/enunciado_lema21.png
:width: 95%

Enunciado: "Ya se demostró $\eta(\boldsymbol{x})$, hay que hacer $1-\eta(\boldsymbol{x})$".
```

Recordemos el resultado de las notas.

```{prf:lemma} Lema 2.1
:label: lem-21

Sea $\mathcal{X}=[0,1]^{d}$, $\mathcal{Y}=\{0,1\}$ y $\mathcal{D}$ una distribución sobre $\mathcal{X}\times\mathcal{Y}$ cuya función de probabilidad condicional $\eta$ es $c$-Lipschitz. Sea $S=(\boldsymbol{x}_1,y_1),\dots,(\boldsymbol{x}_m,y_m)$ una muestra i.i.d., $h_S$ su hipótesis 1-NN y $h^{\star}$ la regla óptima de Bayes para $\eta$. Entonces

$$
\underset{S\sim\mathcal{D}^{m}}{\mathbb{E}}[L_{\mathcal{D}}(h_{S})]\leq 2 L_{\mathcal{D}}(h^{\star})+c\underset{S\sim\mathcal{D}^{m},\, \boldsymbol{x}\sim\mathcal{D}}{\mathbb{E}}[\|\boldsymbol{x}-\boldsymbol{x}_{\pi_{1}(\boldsymbol{x})}\|].
$$
```

El último paso de la demostración necesita la desigualdad elemental

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\;\le\;2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\},\qquad \eta(\boldsymbol{x})\in[0,1],
$$ (eq-desig-min)

que se prueba separando los dos valores posibles del mínimo. Las notas tratan el caso $\min=\eta(\boldsymbol{x})$; **aquí se demuestra el caso $\min=1-\eta(\boldsymbol{x})$** y luego se completa el lema.

Para aligerar la notación, en las Secciones 2.1–2.3 escribimos $t:=\eta(\boldsymbol{x})\in[0,1]$ para un $\boldsymbol{x}$ fijo.

## 2.1 Caso pedido: $\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}=1-\eta(\boldsymbol{x})$

```{prf:proposition} Caso $\min = 1-\eta(\boldsymbol{x})$
:label: prop-caso-1-eta

Si $\eta(\boldsymbol{x})\in[0,1]$ y $\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}=1-\eta(\boldsymbol{x})$, entonces

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\;\le\;2(1-\eta(\boldsymbol{x}))\;=\;2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}.
$$
```

```{prf:proof}
*Hipótesis.* Que el mínimo sea $1-t$ significa que $1-t\le t$, es decir, $t\ge \tfrac12$. Junto con $t\in[0,1]$ tenemos

$$
\tfrac12\le t\le 1\qquad\text{y}\qquad 0\le 1-t\le \tfrac12 .
$$

*Paso 1 (acotar el factor $t$).* Como $t\le1$ y $1-t\ge0$, multiplicar la desigualdad $t\le 1$ por el número no negativo $1-t$ conserva el sentido:

$$
t(1-t)\;\le\;1\cdot(1-t)=1-t .
$$

*Paso 2 (identificar el mínimo).* Por hipótesis, $1-t=\min\{t,1-t\}$.

*Paso 3 (multiplicar por 2).* Multiplicando por $2>0$:

$$
2t(1-t)\;\le\;2(1-t)\;=\;2\min\{t,1-t\}.
$$

Volviendo a $t=\eta(\boldsymbol{x})$ se obtiene el enunciado.

*Caso de igualdad.* $t(1-t)=1-t\iff(1-t)(t-1)=0\iff t=1$. Bajo la hipótesis $t\ge\tfrac12$, la igualdad ocurre solo cuando $\eta(\boldsymbol{x})=1$; por ejemplo, en $t=\tfrac12$ se tiene $2t(1-t)=\tfrac12<1=2\min\{t,1-t\}$.
```

```{admonition} Una forma de escribirlo análoga a la de las notas
:class: note
Con la misma estructura de "cadena" del caso ya resuelto, la versión correcta del caso $\min=1-\eta(\boldsymbol{x})$ es

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\;\underset{\eta(\boldsymbol{x})\le1}{\le}\;2\cdot1\cdot(1-\eta(\boldsymbol{x}))\;=\;2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}.
$$

**No** sirve imitar el paso con cuadrados: aunque $2\eta(1-\eta)\le2\eta^{2}$ es cierto cuando $1-\eta\le\eta$, el paso siguiente, $2\eta^{2}\le2(1-\eta)$, es falso en general. Por ejemplo, con $\eta=1$ diría $2\le0$.
```

## 2.2 Observación sobre el caso $\min=\eta(\boldsymbol{x})$ de las notas

```{admonition} Observación (errata en las notas)
:class: warning
Las notas escriben, para el caso $\min\{\eta,1-\eta\}=\eta$:

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\leq2(1-\eta(\boldsymbol{x}))^{2}\leq 2\min\{\eta(\boldsymbol{x}), 1-\eta(\boldsymbol{x})\}.
$$

La primera desigualdad es correcta (equivale a $\eta\le1-\eta$ multiplicando por $1-\eta\ge0$). La segunda **no**: dice $(1-\eta)^2\le\eta$, que falla, por ejemplo, con $\eta=0$ ($1\le0$) o con $\eta=0.2$ ($0.64\le0.2$). La conclusión sí es cierta; el argumento correcto es simétrico al de la {prf:ref}`prop-caso-1-eta`:

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\;\underset{1-\eta(\boldsymbol{x})\le1}{\le}\;2\eta(\boldsymbol{x})\cdot 1\;=\;2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}.
$$

Es decir: si $\eta\le1-\eta$, se multiplica $1-\eta\le1$ por $\eta\ge0$.
```

```{prf:proof}
**Caso $\min=\eta(\boldsymbol{x})$, corregido.** Si el mínimo es $t$, entonces $t\le1-t$, o sea $0\le t\le\tfrac12$. Como $1-t\le1$ y $t\ge0$, se tiene $t(1-t)\le t\cdot1=t=\min\{t,1-t\}$. Multiplicando por 2 se obtiene {eq}`eq-desig-min`.
```

## 2.3 Una prueba sin casos

Ambos casos son instancias de una identidad: para dos números, el producto es "mínimo por máximo".

```{prf:corollary} Forma unificada
:label: cor-unificada

Para todo $t\in[0,1]$, $\ t(1-t)=\min\{t,1-t\}\cdot\max\{t,1-t\}\le\min\{t,1-t\}$, con igualdad si y solo si $t\in\{0,1\}$.
```

```{prf:proof}
El par $\{t,1-t\}$ consta del mínimo y del máximo, así que su producto es $\min\cdot\max$. Como $\max\{t,1-t\}\le1$ y $\min\{t,1-t\}\ge0$, se tiene $\min\cdot\max\le\min$. Hay igualdad si $\min=0$ (es decir, $t\in\{0,1\}$) o si $\max=1$, que de nuevo significa $t\in\{0,1\}$.
```

## 2.4 Conclusión del Lema 2.1

Con {eq}`eq-desig-min` ya demostrada en ambos casos, se completa la prueba. Escribimos $\boldsymbol{x}':=\boldsymbol{x}_{\pi_1(\boldsymbol{x})}$ e $y':=y_{\pi_1(\boldsymbol{x})}$, como en las notas.

```{prf:proof}
**Lema 2.1, pasos finales.** *(i) Probabilidad de desacuerdo.* Las notas establecen la identidad $(\star)$

$$
\underset{S\sim\mathcal{D}^m}{\mathbb{E}}[L_{\mathcal{D}}(h_S)]=\underset{S_x\sim\mathcal{D}_x^m,\ \boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\Big[\underset{y\sim\eta(\boldsymbol{x}),\ y'\sim\eta(\boldsymbol{x}')}{\mathbb{P}}(y\neq y')\Big].
$$

Dados $\boldsymbol{x},\boldsymbol{x}'$, las etiquetas $y$ e $y'$ son independientes, así que

$$
\mathbb{P}(y\neq y')=\eta(\boldsymbol{x}')(1-\eta(\boldsymbol{x}))+(1-\eta(\boldsymbol{x}'))\,\eta(\boldsymbol{x}).
$$

*(ii) Reescritura.* Desarrollando,

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))+(\eta(\boldsymbol{x})-\eta(\boldsymbol{x}'))(2\eta(\boldsymbol{x})-1)
=\eta(\boldsymbol{x})+\eta(\boldsymbol{x}')-2\eta(\boldsymbol{x})\eta(\boldsymbol{x}'),
$$

que coincide con la expresión de (i), pues $\eta'(1-\eta)+(1-\eta')\eta=\eta+\eta'-2\eta\eta'$.

*(iii) Lipschitz.* Como $|2\eta(\boldsymbol{x})-1|\le1$ y $|\eta(\boldsymbol{x})-\eta(\boldsymbol{x}')|\le c\|\boldsymbol{x}-\boldsymbol{x}'\|$,

$$
\mathbb{P}(y\neq y')\le 2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))+c\,\|\boldsymbol{x}-\boldsymbol{x}'\|.
$$

*(iv) Desigualdad {eq}`eq-desig-min`.* Por la {prf:ref}`prop-caso-1-eta` (caso $\min=1-\eta$) y el caso corregido $\min=\eta$, para **todo** $\boldsymbol{x}$:

$$
2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\le2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}.
$$

*(v) Esperanza.* Tomando esperanza sobre $S_x$ y $\boldsymbol{x}$ (la esperanza es monótona y lineal),

$$
\underset{S\sim\mathcal{D}^m}{\mathbb{E}}[L_{\mathcal{D}}(h_S)]\le 2\,\underset{\boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\big[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}\big]+c\underset{S\sim\mathcal{D}^m,\,\boldsymbol{x}\sim\mathcal{D}_x}{\mathbb{E}}\big[\|\boldsymbol{x}-\boldsymbol{x}_{\pi_1(\boldsymbol{x})}\|\big].
$$

*(vi) Riesgo de Bayes.* Por el {prf:ref}`thm-bayes` del Capítulo 1 (Ecuación {eq}`eq-riesgo-bayes`), $\mathbb{E}_{\boldsymbol{x}}[\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}]=L_{\mathcal{D}}(h^{\star})$. Sustituyendo se obtiene la tesis del Lema 2.1.
```

## 2.5 Verificación numérica

La primera figura compara $2t(1-t)$ con $2\min\{t,1-t\}$ y muestra por qué la cota intermedia $2(1-t)^2$ de las notas no sirve. La segunda celda comprueba el Lema 2.1 con una simulación de 1-NN.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1001)
f, g, wrong = 2*t*(1-t), 2*np.minimum(t, 1-t), 2*(1-t)**2

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, f, lw=2, label=r"$2t(1-t)$")
ax.plot(t, g, lw=2, label=r"$2\min\{t,1-t\}$")
ax.plot(t[t <= .5], wrong[t <= .5], "--", lw=1.5, label=r"$2(1-t)^2$ (cota intermedia de las notas, $t\leq 1/2$)")
ax.axvspan(0.5, 1, color="0.92", label=r"caso $\min=1-t$")
ax.set_xlabel(r"$t=\eta(\mathbf{x})$"); ax.legend(fontsize=8); ax.set_ylim(0, 2.05)
plt.tight_layout(); plt.show()

caso = t >= 0.5
print("Caso min = 1-t: max de 2t(1-t) - 2(1-t) =", (f - 2*(1-t))[caso].max())
print("En [0,1/2]: max de 2(1-t)^2 - 2min =", (wrong - g)[~caso].max(), "(> 0: la cadena de las notas falla)")
```

```{code-cell} ipython3
from sklearn.neighbors import KNeighborsClassifier
rng = np.random.default_rng(1)
d, c = 2, 3.0
# eta con valores en [0.15, 0.85] y gradiente de norma <= c, por lo tanto c-Lipschitz
eta = lambda X: 0.5 + 0.35*np.sin(c/(0.35*np.sqrt(2))*X[:, 0]) * np.cos(X[:, 1])

Xt = rng.uniform(size=(200_000, d)); bayes = np.mean(np.minimum(eta(Xt), 1 - eta(Xt)))
for m in [50, 200, 1000, 5000]:
    err, dist = [], []
    for _ in range(30):
        X = rng.uniform(size=(m, d)); y = (rng.uniform(size=m) < eta(X)).astype(int)
        knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        Xq = rng.uniform(size=(20_000, d))
        dq, _ = knn.kneighbors(Xq, n_neighbors=1)
        # riesgo verdadero exacto de h_S usando eta: E_x[c_h(x)]
        hq = knn.predict(Xq); e = eta(Xq)
        err.append(np.mean(np.where(hq == 1, 1 - e, e))); dist.append(dq.mean())
    cota = 2*bayes + c*np.mean(dist)
    print(f"m={m:5d}  E[L_D(h_S)]≈{np.mean(err):.4f}   cota 2L*+cE||x-x'|| ≈ {cota:.4f}   (L*={bayes:.4f})")
```

En todos los tamaños de muestra, el riesgo de 1-NN queda por debajo de la cota. A medida que $m$ crece, la distancia al vecino más cercano tiende a cero y la cota se acerca a $2L_{\mathcal{D}}(h^{\star})$.
