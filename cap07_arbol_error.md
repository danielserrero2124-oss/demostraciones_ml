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

# 7. Árboles de decisión: el error en la solución del Ejercicio 6

## Enunciado

```{figure} images/enunciado_arbol_a.png
:width: 80%

Sección Random Forest y XGBoost, Ejercicio 6 (solución propuesta en la página de soluciones de las notas).
```

```{figure} images/enunciado_arbol_b.png
:width: 70%

"Buscar dónde está el error en el árbol. No se sabe si el error está aquí o en los otros árboles."
```

**Datos (Tabla 5.1 de las notas).** Diez observaciones; la respuesta es *Gonna Be an Astronaut* (1 = Sí, 0 = No).

| # | Age | Like Dogs | Like Gravity | Astronaut |
|:-:|:---:|:---:|:---:|:---:|
| 1 | 24 | 0 | 0 | 0 |
| 2 | 30 | 1 | 1 | 1 |
| 3 | 36 | 0 | 1 | 1 |
| 4 | 36 | 0 | 0 | 0 |
| 5 | 42 | 0 | 0 | 0 |
| 6 | 44 | 1 | 1 | 1 |
| 7 | 46 | 1 | 0 | 0 |
| 8 | 47 | 1 | 1 | 1 |
| 9 | 47 | 0 | 1 | 1 |
| 10 | 51 | 1 | 1 | 0 |

**Criterio (notas, Sección 5).** Para un nodo $D$ con proporción $p$ de la clase Sí, $\operatorname{Gini}(D)=1-p^{2}-(1-p)^{2}$. El Gini de una partición es $\operatorname{Gini}_{\text{split}}=\frac{N_1}{N}\operatorname{Gini}(D_1)+\frac{N_2}{N}\operatorname{Gini}(D_2)$, y se elige la partición con menor $\operatorname{Gini}_{\text{split}}$.

```{admonition} Respuesta corta
:class: important
El árbol final es **incorrecto**. Hay errores en **varios** de los diagramas, no en uno solo:

1. **Error decisivo (Paso 2, nodo *Like Gravity = Sí*).** Los conteos del umbral *Age* 49 están mal: en realidad $D_1(\text{age}<49)$ tiene **Sí = 5, No = 0** y $D_2(\text{age}\ge 49)$ tiene **Sí = 0, No = 1**. El Gini correcto es **0** (partición perfecta), no 0,26. Como es el menor de todos, el segundo nodo debe ser **Age < 49**, no *Like Dogs*.
2. **Paso 2, *Like Dogs*.** Los conteos también están mal: *Like Dogs = Sí* tiene Sí = 3, No = 1 (la observación 10, de 51 años, sí gusta de los perros y no es astronauta), y *Like Dogs = No* tiene Sí = 2, No = 0. El Gini correcto es $\frac46(0.375)+\frac26(0)=\mathbf{0.25}$, no 0,17.
3. **Paso 1, *Like Dogs* (raíz).** Los conteos son Sí = 3, No = 2 y Sí = 2, No = 3, con Gini 0,48 (no 0,32). El propio diagrama es inconsistente: suma Sí = 4 + 0 = 4, cuando hay 5 astronautas, y usa las fracciones $\frac15,\frac45$ en $D_2$ mientras escribe Sí = 0, No = 5. Este error **no** cambia la raíz, porque *Like Gravity* (0,17) sigue siendo la mejor.
4. **Detalles menores del Paso 1 (*Age*):** en el umbral 49, Sí y No están intercambiados (debe ser $D_1$: Sí = 5, No = 4; $D_2$: Sí = 0, No = 1), aunque el Gini no cambia; en el umbral 43, el peso dice $\frac{8}{10}$ donde debe decir $\frac{5}{10}$ (el resultado 0,48 es correcto); y en 46,5 y 47 el valor exacto es 0,476, que redondea a 0,48.

**Árbol correcto:** *Like Gravity* = No → **No**; *Like Gravity* = Sí → (*Age* < 49 → **Sí**; *Age* ≥ 49 → **No**). Clasifica las 10 observaciones sin error. El árbol propuesto falla en dos: la 9 (47 años, no le gustan los perros, sí es astronauta) y la 10 (51 años, sí le gustan los perros, no es astronauta).
```

A continuación se justifica cada punto con el cálculo completo.

## 7.1 Paso 1: nodo raíz (10 observaciones; 5 Sí, 5 No)

### *Age*

Las edades distintas ordenadas son $24,30,36,42,44,46,47,51$. Los umbrales del punto medio, $(x_{(i)}+x_{(i+1)})/2$, son $27,33,39,43,45,46.5,49$. La solución propuesta incluye además $36$ y $47$, que son puntos medios de valores **repetidos** (36 y 36, 47 y 47). Con la regla "<", la partición de $36$ coincide con la de $33$ y la de $47$ con la de $46.5$; no es un error, pero esos umbrales son redundantes.

```{code-cell} ipython3
import numpy as np, pandas as pd
df = pd.DataFrame({
    "Age":         [24, 30, 36, 36, 42, 44, 46, 47, 47, 51],
    "LikeDogs":    [ 0,  1,  0,  0,  0,  1,  1,  1,  0,  1],
    "LikeGravity": [ 0,  1,  1,  0,  0,  1,  0,  1,  1,  1],
    "Astronaut":   [ 0,  1,  1,  0,  0,  1,  0,  1,  1,  0]}, index=range(1, 11))

def gini(y):
    if len(y) == 0: return 0.0
    p = np.mean(y); return 1 - p**2 - (1 - p)**2

def evalua(d, var, umbral, etiquetas=("D1", "D2")):
    """D1: var < umbral ; D2: var >= umbral."""
    m = d[var] < umbral
    fila = {}
    for et, g in zip(etiquetas, (d[m], d[~m])):
        fila[f"{et}: N"] = len(g); fila[f"{et}: Sí"] = int(g.Astronaut.sum()); fila[f"{et}: No"] = int((1 - g.Astronaut).sum())
        fila[f"Gini({et})"] = round(gini(g.Astronaut), 4)
    fila["Gini split"] = round((m.sum()*gini(d[m].Astronaut) + (~m).sum()*gini(d[~m].Astronaut))/len(d), 4)
    return fila

umbrales = [27, 33, 36, 39, 43, 45, 46.5, 47, 49]
tabla_age = pd.DataFrame({u: evalua(df, "Age", u) for u in umbrales}).T
tabla_age.index.name = "Umbral Age"
tabla_age
```

Se comparan los conteos con los diagramas de la solución propuesta:

```{figure} images/TREEDECISION1.png
:width: 75%

Solución propuesta, Paso 1, *Age*. Los conteos coinciden con la tabla, salvo en el umbral 49 (Sí y No intercambiados: la observación 10, de 51 años, es **No**) y en el peso $\frac{8}{10}$ del umbral 43.
```

El mínimo es $0.4444$, empatado entre los umbrales 27 y 49. La solución elige 27 ("el primero que alcance dicho valor").

### *Like Dogs* y *Like Gravity*

```{code-cell} ipython3
tabla_bin = pd.DataFrame({v: evalua(df, v, 0.5, ("No (0)", "Sí (1)")) for v in ["LikeDogs", "LikeGravity"]}).T
tabla_bin
```

```{figure} images/TREEDECISION2.png
:width: 80%

Solución propuesta, Paso 1, *Like Dogs*: **incorrecta**. Dice $D_1(\text{Sí})$: Sí = 4, No = 1, y $D_2(\text{No})$: Sí = 0, No = 5. Lo correcto es Sí = 3, No = 2 y Sí = 2, No = 3, con Gini 0,48 en ambos nodos y en la partición.
```

Comprobación manual de *Like Dogs*. Las observaciones con *Like Dogs* = 1 son {2, 6, 7, 8, 10}, con respuestas (1, 1, 0, 1, 0): Sí = 3, No = 2. Las de *Like Dogs* = 0 son {1, 3, 4, 5, 9}, con respuestas (0, 1, 0, 0, 1): Sí = 2, No = 3. Entonces

$$
\begin{gathered}
\operatorname{Gini}(D_{\text{Sí}})=1-\left(\tfrac35\right)^{2}-\left(\tfrac25\right)^{2}=\tfrac{12}{25}=0.48,\\
\operatorname{Gini}(D_{\text{No}})=1-\left(\tfrac25\right)^{2}-\left(\tfrac35\right)^{2}=0.48,\\
\operatorname{Gini}_{\text{split}}=0.48 .
\end{gathered}
$$

```{figure} images/TREEDECISION3.png
:width: 80%

Solución propuesta, Paso 1, *Like Gravity*: **correcta**. Sí → 6 obs. (5 Sí, 1 No), Gini $=\tfrac{10}{36}=0.278$; No → 4 obs. (0 Sí, 4 No), Gini $=0$; partición $=\tfrac{6}{10}\cdot0.278=0.167$.
```

**Conclusión del Paso 1:** la raíz correcta es *Like Gravity* (0,167 < 0,444 < 0,48). La rama *No* es una hoja pura (No). El error de *Like Dogs* no altera esta decisión.

## 7.2 Paso 2: nodo *Like Gravity* = Sí (6 observaciones)

El subconjunto es el de la tabla de la solución propuesta, que es correcta:

```{code-cell} ipython3
s = df[df.LikeGravity == 1]
s
```

Las edades distintas son $30,36,44,47,51$, con umbrales $33,40,45.5,49$ (más $47$, redundante, que repite la partición de $45.5$).

```{code-cell} ipython3
tabla2 = pd.DataFrame({u: evalua(s, "Age", u) for u in [33, 40, 45.5, 47, 49]}).T
tabla2.index.name = "Umbral Age"
tabla2.loc["LikeDogs (No | Sí)"] = pd.Series(evalua(s, "LikeDogs", 0.5)).values
tabla2
```

```{figure} images/TREEDECISION4.png
:width: 75%

Solución propuesta, Paso 2, *Age*. Los umbrales 33, 40, 45,5 y 47 son correctos. **El umbral 49 es incorrecto:** dice $D_1$: Sí = 4, No = 1 y $D_2$: Sí = 1, No = 0.
```

Comprobación manual del umbral 49. Con *Age* < 49 quedan las observaciones {2, 3, 6, 8, 9} (edades 30, 36, 44, 47, 47), **todas astronautas**: Sí = 5, No = 0. Con *Age* ≥ 49 queda la observación {10} (51 años), que **no** es astronauta: Sí = 0, No = 1. Por lo tanto,

$$
\begin{gathered}
\operatorname{Gini}(D_1)=1-1^{2}-0^{2}=0,\\
\operatorname{Gini}(D_2)=1-0^{2}-1^{2}=0,\\
\operatorname{Gini}_{\text{split}}=\tfrac56\cdot0+\tfrac16\cdot0=\mathbf{0}.
\end{gathered}
$$

```{figure} images/TREEDECISION5.png
:width: 80%

Solución propuesta, Paso 2, *Like Dogs*: **incorrecta**. Dice $D_1(\text{Sí})$: Sí = 4, No = 0 y $D_2(\text{No})$: Sí = 1, No = 1.
```

Comprobación manual de *Like Dogs* en el subconjunto. *Like Dogs* = 1 corresponde a las observaciones {2, 6, 8, 10}, con respuestas (1, 1, 1, **0**): Sí = 3, No = 1. *Like Dogs* = 0 corresponde a {3, 9}, con respuestas (1, 1): Sí = 2, No = 0. Así,

$$
\begin{gathered}
\operatorname{Gini}(D_{\text{Sí}})=1-\left(\tfrac34\right)^{2}-\left(\tfrac14\right)^{2}=\tfrac{6}{16}=0.375,\\
\operatorname{Gini}(D_{\text{No}})=0,\\
\operatorname{Gini}_{\text{split}}=\tfrac46(0.375)=\mathbf{0.25}.
\end{gathered}
$$

El único "No" del subconjunto (la observación 10) tiene *Like Dogs* = 1, así que no puede haber un nodo *Like Dogs = Sí* puro.

**Conclusión del Paso 2:** el mínimo es **Age < 49 con Gini 0**, no *Like Dogs* (que vale 0,25, empatado con el umbral 40). Ambos hijos son puros, de modo que el árbol termina aquí.

## 7.3 Paso 3: el árbol propuesto frente al árbol correcto

```{figure} images/TREEDECISION6.png
:width: 70%

Árbol propuesto en la solución.
```

Además de partir de valores de Gini erróneos, el Paso 3 tiene un problema de procedimiento: el umbral 45,5 se calculó con las **6** observaciones del Paso 2, pero se aplica al nodo *Like Dogs = No*, que contiene solo las observaciones {3, 9}. Ese nodo ya es puro (ambas Sí), así que no necesita otra división. Por eso el umbral 45,5 envía la observación 9 (47 años, astronauta) a la hoja *No*. A la vez, el nodo *Like Dogs = Sí* se declaró hoja "Sí" cuando contiene un "No" (la observación 10).

```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

def arbol_propuesto(r):
    if r.LikeGravity == 0: return 0
    if r.LikeDogs == 1:    return 1
    return 1 if r.Age < 45.5 else 0

def arbol_correcto(r):
    if r.LikeGravity == 0: return 0
    return 1 if r.Age < 49 else 0

res = df.assign(propuesto=df.apply(arbol_propuesto, axis=1), correcto=df.apply(arbol_correcto, axis=1))
print("Exactitud de entrenamiento -> propuesto:", (res.propuesto == res.Astronaut).mean(),
      "| correcto:", (res.correcto == res.Astronaut).mean())
print("Observaciones mal clasificadas por el árbol propuesto:")
display(res[res.propuesto != res.Astronaut])

# Verificación independiente con scikit-learn (criterio Gini, CART)
clf = DecisionTreeClassifier(criterion="gini", random_state=0).fit(df[["Age", "LikeDogs", "LikeGravity"]], df.Astronaut)
print(export_text(clf, feature_names=["Age", "LikeDogs", "LikeGravity"]))
fig, ax = plt.subplots(figsize=(7, 4))
plot_tree(clf, feature_names=["Age", "LikeDogs", "LikeGravity"], class_names=["No", "Sí"], filled=True, impurity=True, ax=ax)
plt.show()
```

`scikit-learn` construye el mismo árbol correcto: raíz *LikeGravity* ≤ 0,5 y, en la rama Sí, *Age* ≤ 49. Usa "≤" con el umbral en el punto medio, lo que equivale a "< 49" para estos datos.

```{admonition} Árbol correcto
:class: tip
- Si **no** te gusta la gravedad → **no** serás astronauta (4 obs., Gini 0).
- Si te gusta la gravedad y tienes **menos de 49 años** → **sí** serás astronauta (5 obs., Gini 0).
- Si te gusta la gravedad y tienes **49 años o más** → **no** serás astronauta (1 obs., Gini 0).

*Like Dogs* no aparece: no aporta reducción de impureza adicional.
```
