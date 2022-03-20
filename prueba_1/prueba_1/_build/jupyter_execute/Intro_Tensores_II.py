#!/usr/bin/env python
# coding: utf-8

# # <span style="color:#F72585"><center>Tensores</center></span>

# <center>Introducción</center>

# <figure>
# <center>
# <img src="producto_tensorial.png" width="600" height="300" /> 
# </center>
# </figure>
# 
# Fuente: Alvaro Montenegro

# ## <span style="color:#4361EE">Referencias</span>

# 1. [Alvaro Montenegro y Daniel Montenegro, Inteligencia Artificial y Aprendizaje Profundo, 2021](https://github.com/AprendizajeProfundo/Diplomado)
# 1. [Alvaro Montenegro, Daniel Montenegro y Oleg Jarma, Inteligencia Artificial y Aprendizaje Profundo Avanzado, 2022](https://github.com/AprendizajeProfundo/Diplomado-Avanzado)

# ## <span style="color:#4361EE">Introducción</span>

# En esta lección aprenderemos los conceptos básicos de tensores y como los usamos para manipular imágenes usando tensores.

# ## <span style="color:#4361EE">Tensor</span>

# Un tensor es un concepto matemático que generaliza los conceptos de escalares, vectores y matrices.

# <figure>
# <center>
# <img src="../Imagenes/tensores.png" width="600" height="300" /> 
# </center>
# </figure>
# 
# Fuente: Alvaro Montenegro

# En términos muy simples, un tensor es un objeto dinámico (*matemáticamente diríamos que es una función entre espacios vectoriales*) que vive dentro de una estructura. 
# 
# Pero no vamos a hacer un tratado matemático aquí. 
# 
# Lo importante en esta clase es entender que en realidad, escalares, vectores, matrices pueden verse como tensores fijos y eso será suficiente para lo que sigue.

# ## <span style="color:#4361EE">Rango</span>

# Diremos que los escalares tienen rango (*shape*) 0, los vectores tiene rango 1, las matrices rango 2 y el tensor de la derecha rango 3. 
# 
# El rango corresponde al número de índices que se requiere para identificar de manera única a cada elemento del tensor.
# 
# Observe que por ejemplo, en el último tensor, requiere (fila, columna, cajón). 
# 
# También podría ser (cajón, fila, columna).

# ## <span style="color:#4361EE">Redes Neuronales</span>

# La siguiente imagen muestra el estado en un instante de una una parte oculta de una red  neuronal profunda.

# <figure>
# <center>
# <img src="../Imagenes/fragmento_red.png" width="500" height="300" /> 
# </center>
# </figure>
# 
# Fuente: Alvaro Montenegro

# El proceso puede modelarse en forma simplificada usando matrices y vectores como se ve a continuación.

# $$
# W_{12}L_1 = L2 \to \begin{pmatrix} -1 & 0.4 & 1.5\\ 0.8 & 0.5 & 0.75 \\ 0.2 & -0.3  & 1\\ \end{pmatrix}\begin{pmatrix} 2.5\\ 4  \\ 1.2 \end{pmatrix} = \begin{pmatrix} 0.9\\ 4.9 \\ 0.5 \end{pmatrix}
# $$

# Observe por ejemplo que $$-1\times 2.5 + 0.4\times 4  + 1.5\times 1.2 = 0.9$$

# En la fase de entrenamiento de la red neuronal, los pesos de la matriz se van modificando hasta que se encuentra un óptimo local. Este proceso ocurre en toda la estructura de la red.
# 
# Por lo que no parece extraño que las GPU y las  TPU pasen todo el tiempo haciendo operaciones de este tipo, que al final se reduce a sumas y multiplicaciones.
# 
# Por otro lado, lo que ocurre es que los objetos que se procesan no necesariamente son vectores como en el ejemplo, y esto lleva a la necesidad de generalizar los conceptos.
# 

# ## <span style="color:#4361EE">Producto tensorial</span>

# La operación más ejecutada en aprendizaje profundo es el producto tensorial.
# 
# Vamos a suponer que cada elemento en los tensores de rango 3 se indexan mediante coordenadas (fila, columna, profundidad) y que los tensores de rango 2 se indexan como (fila, columna).
# 
# La siguiente imagen ilustra la forma de un producto tensorial. 
# 
# - A la izquierda (azul) se tiene un tensor de tamaño digamos $n \times p \times s$. 
# 
# - El tensor que está operando en el centro (rosa) es  de tamaño $p \times r$. Este actúa operando en este caso sobre cada capa del tensor de la izquierda haciendo un producto usual de matrices. 
# 
# - Por lo que el tensor resultante (verde) a la derecha tiene tamaño $n \times r \times s$
# 
# 

# <figure>
# <center>
# <img src="../Imagenes/producto_tensorial.png" width="600" height="300" /> 
# </center>
# </figure>
# 
# Fuente: Alvaro Montenegro

# ## <span style="color:#4361EE">Explicación del producto</span>

# La explicación del proceso es la siguiente:
# 
# Cada capa frontal del tensor azul es multiplica por el tensor rosa y el resultado es colocando como una capa frontal en el tensor resultante (verde).
# 
# Cada multiplicación es entre dos matrices (azul * rosa) y el resultado es una matriz (verde).
# 
# Cada multiplicación de matrices se hace por la fórmula fila (matriz azul) * columna (matriz rosa)
# 
# Vamos por ejemplo a suponer que una capa roja es $ azul  = \begin{pmatrix} 1 & 2 & 1\\ 3 & 4 & 1 \\ 4 & 5  & 0\\ \end{pmatrix}$, $rosa = \begin{pmatrix} 5 & 10\\ 20 & 30 \\ 4 & 1\end{pmatrix}$
# 
# Entonces se tiene que 
# 
# $$
# azul \times rosa = \begin{pmatrix} 1 & 2 & 1\\ 3 & 4 & 1 \\ 4 & 5 & 0\\ \end{pmatrix} \times \begin{pmatrix} 5 & 10\\ 20 & 30\\ 4 & 1\end{pmatrix} = \begin{pmatrix} 1\times 5 + 2 \times 20 +  1 \times 4 & 1 \times 10 + 2\times 30 + 1\times 1
# \\ 3\times 5 + 4 \times 20 + 1 \times 4 & 3 \times 10 + 4 \times 30 + 1 \times 1
# \\ 4\times 5 + 5 \times 20 + 0 \times 4 & 4 \times 10 + 5 \times 30 + 0 \times 1\end{pmatrix} = turquesa
# $$

# ## <span style="color:#4361EE">Imágenes a color</span>

# De manera clásica una imagen a color está compuesta de tres colores primarios: rojo (*Red*), verde (*Green*) y azul (*Blue*). Para generar una imagen a color un computador maneja tres planos de color, los cuales son controlados desde tensores tridimensionales. Considere el siguiente ejemplo.

# <figure>
# <center>
# <img src="../Imagenes/zeus_2.png" width="600" height="300" /> 
# </center>
# </figure>
# 
# Fuente: Alvaro Montenegro

# Cada pixel (*punto*) de la imagen es representado por una valor numérico en el rango de 0 a 255, o en rango de valores reales entre cero y 1.

# ## <span style="color:#4361EE">Construcción aleatoria de una imagen</span>

# Considere el siguiente código Python.

# In[1]:


import numpy as np
I=np.random.randint(0,255,size=(3,10,10))
print(I)


# Este tensor representa una imagen de tamaño $10 \times 10$. Son tres planos de color $10 \times 10$.
# 
# Observe que la primera dimensión corresponde a cada plano de color y las restantes dos dimensiones a las intensidades de cada color para cada punto.
# 
# Renderizar (dibujar en este caso), nos lleva a la siguiente imagen.

# In[2]:


# conda install -c conda-forge matplotlib
import matplotlib.pyplot as plt

plt.imshow(I.T)
plt.show()


# Observe que 

# In[3]:


(I.T).shape


# Porque Python maneja las imágenes en este formato: Fila, columna y plano de color.

# ## <span style="color:#4361EE">Imagen real</span>

# Vamos a trabajar ahora con una imagen real.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# conda install -c anaconda scikit-image
from skimage import data
from skimage.color import rgb2gray

original = data.astronaut()
grayscale = rgb2gray(original)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()


# In[5]:


Idata=np.array(grayscale)
print("\nLa imagen tiene forma: ",Idata.shape,"\n")
print(Idata)


# ## <span style="color:#4361EE">Planos de color</span>

# In[6]:


Idata = np.array(original)
print("\nLa imagen tiene forma: ",Idata.shape,"\n")
print("\nEscala de Rojos:\n\n",Idata[:511,:511,0],"\n")
print("\nEscala de Verdes:\n\n",Idata[:511,:511,1],"\n")
print("\nEscala de Azules:\n\n",Idata[:511,:511,2],"\n")


# In[7]:


fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(15,15))

ax1.imshow(Idata[:,:,0],cmap="Reds")
ax1.set_xlabel('Red')
ax2.imshow(Idata[:,:,1],cmap="Greens")
ax2.set_xlabel('Green')
ax3.imshow(Idata[:,:,2],cmap="Blues")
ax3.set_xlabel('Blue')
plt.show()


# ## <span style="color:#4361EE">Manipulación  de imágenes</span>

# ### <span style="color:#4CC9F0">Intercambia dos planos de color</span>

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray

original = data.astronaut()

Idata_m = Idata
Idata_m[:,:,0], Idata_m[:,:,2] = Idata_m[:,:,2], Idata_m[:,:,0]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(Idata_m)
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# ### <span style="color:#4CC9F0">Suma una constante a la imagen</span>

# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

k = 10
ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(original + k)
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# In[10]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

k = 2
ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(original //k)
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# In[11]:


Idata_m = Idata

Idata_m[:,:,0 ]=0

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(Idata_m)
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# In[12]:


Idata


# In[13]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(255 - Idata)
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# ## <span style="color:#4361EE">Colocar dos imagenes en un tensor</span>

# Esta es una forma para organizar conjuntos de imágenes en un único tensor

# In[14]:


original= np.expand_dims(original,axis=0)


# In[15]:


original.shape


# In[16]:


Idata_m= np.expand_dims(Idata_m,axis=0)


# In[17]:


images = np.concatenate((original, Idata_m),axis=0)


# In[18]:


images.shape


# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(images[0])
ax[0].set_title("Original")
ax[1].imshow(images[1])
ax[1].set_title("Modificada")

fig.tight_layout()
plt.show()


# ## <span style="color:#4361EE">Trasformaciones afines</span>

# En este ejemplo usaremos la librería OpenCV.
# 
# Esta es la imagen original tomada de [omes-va.com](https://omes-va.com/trasladar-rotar-escalar-recortar-una-imagen-opencv/). Código tomado del mismo sitio.

# In[20]:


import numpy as np
import cv2
image = cv2.imread('../Imagenes/ave.jpeg')
cv2.imshow('Imagen de entrada',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Voy aquí


# ## <span style="color:#4361EE">Translación</span>

# $$
# M =\begin{pmatrix} 1 & 0 & Tx\\
# 0 & 1 & Ty
# \end{pmatrix}
# $$
# 
# - Tx, representa el desplazamiento en x.
# 
# - Ty, representa el desplazamiento en y.
# 
# 

# In[ ]:


# Translación
ancho = image.shape[1] #columnas
alto = image.shape[0] # filas
# Traslación
M = np.float32([[1,0,100],[0,1,150]])
imageOut = cv2.warpAffine(image,M,(ancho,alto))
cv2.imshow('Imagen de entrada',image)
cv2.imshow('Imagen de salida',imageOut)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## <span style="color:#4361EE">Rotación</span>

# $$
# M =\begin{pmatrix} \cos \theta & -\sin \theta & 0\\
# \cos \theta & \sin \theta & 1\\
# 0 & 0  & 1
# \end{pmatrix}
# $$
# 
# - $\theta$ representa el ángulo de rotación. En este ejemplo $\theta = \pi/4$ 0 lo que es lo mismo $45^o$.

# In[ ]:


# rotación
ancho = image.shape[1] #columnas
alto = image.shape[0] # filas

M = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
imageOut = cv2.warpAffine(image,M,(ancho,alto))
cv2.imshow('Imagen de entrada',image)
cv2.imshow('Imagen de salida',imageOut)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


image.shape

