# Demostraciones de la tarea — Machine Learning

<div class="portada">
  <img src="_static/autor.png" alt="Daniel Serrano Romero" class="portada-foto">
  <div class="portada-texto">
    <p class="portada-autor">Daniel Serrano Romero</p>
    <p class="portada-detalle">Machine Learning · Universidad del Norte</p>
    <p class="portada-detalle">Octubre de 2026</p>
  </div>
  <img src="_static/logo_uninorte.png" alt="Universidad del Norte" class="portada-logo">
</div>

Este libro reúne, desarrolladas paso a paso, las demostraciones y verificaciones solicitadas en el documento *Demostraciones_Tarea*. Todas siguen las **notaciones de las notas de clase** {cite}`rubio_ml` (definiciones, lemas y numeración de ecuaciones como (2.2), (3.8), (3.11), (3.12), (5.4)–(5.7), Algoritmo 5.1, etc.), de modo que cada resultado puede leerse al lado del texto original.

Cada capítulo tiene la misma estructura:

1. **Enunciado**: la captura del documento de la tarea y la formulación precisa de lo que se demuestra.
2. **Demostración**: el argumento completo, sin omitir pasos algebraicos.
3. **Verificación numérica**: una celda de Python (ejecutada al construir el libro) que comprueba el resultado con datos simulados o reales. No sustituye a la demostración: sirve como control de errores.

```{list-table} Contenido
:header-rows: 1
:widths: 8 22 70

* - Cap.
  - Tema
  - Qué se demuestra
* - 1
  - K-NN
  - Definición 2.3: equivalencias del riesgo verdadero, descomposición mediante $\eta(\boldsymbol{x})$, insesgamiento del riesgo empírico y optimalidad del predictor de Bayes.
* - 2
  - K-NN
  - Lema 2.1: la desigualdad $2\eta(\boldsymbol{x})(1-\eta(\boldsymbol{x}))\le 2\min\{\eta(\boldsymbol{x}),1-\eta(\boldsymbol{x})\}$ en el caso $\min=1-\eta(\boldsymbol{x})$ (y una corrección del caso $\min=\eta(\boldsymbol{x})$).
* - 3
  - Ridge
  - Las cuatro propiedades (3.12) del estimador ridge: esperanza, covarianza, sesgo y error cuadrático medio.
* - 4
  - Ridge
  - $Z^{T}\boldsymbol{1}=\boldsymbol{1}^{T}Z=0$ y su papel en las ecuaciones normales (3.8) y (3.10).
* - 5
  - Ridge
  - Las igualdades (3.11) tras estandarizar, y la consecuencia $\beta_0^{\star}=0$.
* - 6
  - Lasso
  - Estimación de los parámetros lasso: condiciones de optimalidad, solución cerrada (umbralización suave), descenso por coordenadas y algoritmo LARS.
* - 7
  - Árboles
  - Localización y corrección del error en la solución propuesta del Ejercicio 6 (Sección Random Forest y XGBoost).
* - 8
  - Boosting
  - Derivación completa del Algoritmo 5.1 (AdaBoost) y la cota exponencial del error de entrenamiento.
```

```{admonition} Observaciones sobre las notas
:class: tip
Al desarrollar las demostraciones se encontraron algunas erratas en las notas originales; cada una se señala en el capítulo correspondiente con un recuadro de **Observación**. Las más relevantes son:

- En el Lema 2.1, la cadena $2\eta(1-\eta)\le 2(1-\eta)^2\le 2\min\{\eta,1-\eta\}$ no es válida en general (el segundo $\le$ falla, por ejemplo, con $\eta=0$). La conclusión sí es cierta, y en el {doc}`cap02_knn_lema_1nn` se da un argumento correcto para ambos casos.
- En el árbol del Ejercicio 6, los conteos de la variable *Like Dogs* (en la raíz y en el segundo nivel) y los del umbral 49 del segundo nivel están mal; por eso el árbol final es incorrecto. Ver {doc}`cap07_arbol_error`.
```

## Cómo construir el libro

```bash
pip install "jupyter-book<2" sphinx-proof numpy pandas matplotlib scikit-learn
jupyter-book build .
# el resultado queda en _build/html/index.html
```
