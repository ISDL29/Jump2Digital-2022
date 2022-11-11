# background
## Clasificación de la calidad del aire
El Acuerdo de París es un tratado internacional sobre el cambio climático que fue adoptado por 196 Partes en la COP21 de París. Su objetivo es limitar el calentamiento mundial muy por debajo de 2, preferiblemente a 1,5 grados centígrados, en comparación con los niveles preindustriales.

Para alcanzar este objetivo de temperatura a largo plazo, los países se proponen alcanzar el máximo de las emisiones de gases de efecto invernadero lo antes posible para lograr un planeta con clima neutro para mediados de siglo.

Es por ello que la Unión Europea esta destinando grandes cantidades de recursos al desarrollo de nuevas tecnologías que permitan la mejorar la lucha contra la contaminación. Una de estas es un nuevo tipo de sensor basado en tecnología láser que permita detectar la calidad del aire en función de diferentes mediciones.

# problem

## Predicción de la calidad del aire

El sensor mide 8 características del aire. A partir de esas características, se tiene que determinar la calidad del aire, clasificada en 3 categorías (buena mala y regular).

El dataset consta de 2 archivos:

- **train.csv**: Este dataset contiene tanto las variables predictoras como el tipo de clasificación de calidad del aire.

- **test.csv**: Este dataset contiene las variables predictoras con las que se tendrá que predecir el tipo de calidad de aire.

La estructura del dataset es la siguiente:

- **Features**: El dataset contiene 8 features en 8 columnas que son los parámetros medidos por los diferentes sensores. Estos corresponden a las diferentes interacciones que han tenido los haces de los láseres al travesar las partículas del aire.

- **Target**: El target corresponde al 'label' que clasifica la calidad del aire.

  - Target **0** corresponde a una calidad del aire **Buena**
  - Target **1** corresponde a una calidad del aire **Moderada**
  - Target **2** corresponde a una calidad del aire **Peligrosa**

# results

Ver Jump2Digital2022.ipynb

Los gráficos son interactivos, así que se necesita un entorno local con plotly instalado para visualizarlos.

# analysis

Ver Jump2Digital2022.ipynb

Los gráficos son interactivos, así que se necesita un entorno local con plotly instalado para visualizarlos.

# solution

Ver Jump2Digital2022.ipynb

Los gráficos son interactivos, así que se necesita un entorno local con plotly instalado para visualizarlos.

# license
