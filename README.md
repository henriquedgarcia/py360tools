# py360tools
## Sumário

- [Introdução](#introdução)
- [API](#api) 
- [Exemplos](#examples)
- [Outras Funções](#other-functions)
- [Dependências](#depends)
- [Instalação](#install)

## Introdução

O py360tools é um conjunto de ferramentas para manipulação de vídeo 360° com 
ladrilhos (tiles). Sua arquitetura se baseia nas transformações do domínio da 
projeção no domíno da esfera e vice-versa. Com uma interface comum é possível 
que um pixel transite para qualquer projeção/viewport.

<img src="docs/img/arquitetura.png" alt="Descrição" width="300"/>

As classes principais são as classes das projeções. Atualmente a py360tools 
suporta apenas projeção equirretangular e projeção cubemap. A projeção cria 
"Tiles" que contem informações que serão usadas por outras classes. A projeção 
é injetada no Viewport para podermos deter os ladrilhos (Tiles) que estão sendo 
vistos e para extrair o viewport. Com a lista de ladrilhos vistos e sabendo a 
URL para seus respectivos arquivos de vídeo, a classe TileStitcher gerencia e
sincroniza o fluxo de quadros de todos os ladrilhos a fim de reconstruir a projeção 
com apenas os ladrilhos selecionados. Em seguida, esta projeção pode ser 
repassada para classe Viewport para se extrair o que é visto pelo usuário.

<img src="docs/img/classes.png" alt="Descrição" width="300"/>

Para a detecção de visibilidade, consideramos que o campo de visão do usuário é
uma pirâmide de base quadrada com o topo saindo dos olhos do usuário. Cada face 
da pirâmide pode ser modelada por quatro planos e cada plano pode ser definido 
pelas usa sua normal. Um usuário com FOV de 120°x90° em repouso olhando para 
frente na posição yaw=0, pitch=0 e roll=0 possui as seguintes normais:

$$ 
n_1 = (x_1, y_1, z_1) = \left(-\cos(\frac{FOV_x}{2}+90°), 0, -\sin(\frac{FOV_x}{2}+90) \right) 
$$

$$ 
n_2 = (x_2, y_2, z_2) = \left(\cos(\frac{FOV_x}{2}+90°), 0, -\sin(\frac{FOV_x}{2}+90) \right)
$$

$$ 
n_3 = (x_3, y_3, z_3) = \left(0, -\cos(\frac{FOV_y}{2}+90°), -\sin(\frac{FOV_y}{2}+90°) \right) 
$$

$$ 
n_4 = (x_4, y_4, z_4) = \left(0, \cos(\frac{FOV_y}{2}+90°), -\sin(\frac{FOV_y}{2}+90°) \right) 
$$

O viewport é a seção da esfera que
está dentro da pirâmide. Considerando que as quatro normais estão apontando 
para fora da pirâmide, um ponto $ \vec{p}=(x, y, z) $ é considerado 
dentro da pirâmide se satisfazer as quatro equações abaixo:

$$ 
x_1 \times x + y_1 \times y + z_1 \times z < 0
$$

$$
x_2 \times x + y_2 \times y + z_2 \times z < 0 
$$

$$
x_3 \times x + y_3 \times y + z_3 \times z < 0 
$$

$$
x_4 \times x + y_4 \times y + z_4 \times z < 0 
$$

Por questões de eficiência, não necessário testar todos os pixels de um ladriho 
para verificar se ele está sendo visto, basta testar as bordas. Já vi alguns 
trabalhos selecionarem vários pontos dentro do ladrilho, mas acho isso impreciso 
e só justificado se o sistema estiver muito lento.

<img src="docs/img/get_tiles.png" alt="Descrição" width="300"/>

Para extrair o viewport de uma projeção partimos de uma imagem em branco. 
Convertemos as coordenadas dos pixels do viewport (0, 0), ..., (0, m), ..., 
(n, 0), ... (n, m) em coordenadas cartesianas (x, y, z). Em seguida de coordenadas 
cartesianas para o domíno da projeção que contem os ladrilhos. Usando a função 
de [remap do OpenCv](https://docs.opencv.org/4.x/d1/da0/tutorial_remap.html)
remapeamos os pixels da projeção de volta para o viewport. 

<img src="docs/img/get_viewport.png" alt="Descrição" width="300"/>

Converter de uma projeção em outra o processo é o mesmo. Partimos de uma projeção
em branco e vamos buscar os valores dos pixels em outra projeção.

<img src="docs/img/convert_projection.png" alt="Descrição" width="300"/>

Classe para remontar o fluxo de streaming do stiles em projeção. Ela gerencia os múltiplos fluxos dos ladrilhos sincronizando todos. 

<img src="docs/img/tile_stitcher.png" alt="Descrição" width="300"/>

## API

As projeções são subclasses de
`py360tools.assets.projection_base.ProjectionBase`

```python
class Viewport:
    def __init__(self, resolution, fov, projection: ProjectionBase = None):
        ...
```

**Parameters**

- `resolution`: _str_ <br>
  &nbsp;&nbsp;&nbsp;&nbsp; A string representing the viewport resolution, e.g.,
  '600x3000'.
- `fov`: _str_ <br>
  &nbsp;&nbsp;&nbsp;&nbsp; A string representing the field of view (FOV), e.g.,
  '100x90'.
- `projection`: _ProjectionBase_ <br>
  &nbsp;&nbsp;&nbsp;&nbsp; [Optional] A object representation the projection,
  e.g., ERP, CMP.

**Methods**

- `extract_viewport(frame_array, yaw_pitch_roll=None) -> np.ndarray`: shape==(2,...)<br>
  &nbsp;&nbsp;&nbsp;&nbsp; Extracts a viewport image based on the input frame and
  yaw-pitch-roll orientation.
- `get_vptiles(yaw_pitch_roll=None) -> list[int]`: <br>
  &nbsp;&nbsp;&nbsp;&nbsp; Get the tiles used in the viewport.
- `is_viewport(yaw_pitch_roll=None) -> bool`: <br>
  &nbsp;&nbsp;&nbsp;&nbsp; Verify if the given set of 3D points is within the viewport
- `get_vp_mask(lum=255) -> bool`: <br>
  &nbsp;&nbsp;&nbsp;&nbsp; Verify if the given set of 3D points is within the viewport
- `is_viewport(yaw_pitch_roll=None) -> bool`: <br>
  &nbsp;&nbsp;&nbsp;&nbsp; Verify if the given set of 3D points is within the viewport
- `is_viewport(yaw_pitch_roll=None) -> bool`: <br>
  &nbsp;&nbsp;&nbsp;&nbsp; Verify if the given set of 3D points is within the viewport

**Attributes**

- `yaw_pitch_roll`: np.ndarray<br>
  &nbsp;&nbsp;&nbsp;&nbsp; Coordinates in the projection plane with shape==(2, ...).
- `xyz`: np.ndarray<br>
  &nbsp;&nbsp;&nbsp;&nbsp; Coordinates on the 3d cartesian space with shape==(3, ...)

```python
class ProjectionBase:
    def __init__(self, proj_res, fov):
        ...
```

**Parameters**

- `proj_res`: _str_ <br>
  &nbsp;&nbsp;&nbsp;&nbsp; A string representing the projection resolution, e.g., '600x3000'.
- `fov`: _str_ <br>
  &nbsp;&nbsp;&nbsp;&nbsp; A string representing the field of view (FOV), e.g., '100x90'.

**Methods**

- `xyz2nm(xyz) -> np.ndarray`: xyz.shape==(3,...)<br>
  &nbsp;&nbsp;&nbsp;&nbsp; projects any point in space onto a projection. Returns
  the coordinates of the projection.
- `nm2xyz(nm) -> np.ndarray`: nm.shape==(2,...)<br>
  &nbsp;&nbsp;&nbsp;&nbsp; The projection coordinate system is the same as
  the image. It starts at the top left and increases to the right (m) and down (n).
  Returns the coordinates for the surface of the 3D sphere.

**Attributes**

- `nm`: np.ndarray<br>
  &nbsp;&nbsp;&nbsp;&nbsp; Coordinates in the projection plane with shape==(2, ...).
- `xyz`: np.ndarray<br>
  &nbsp;&nbsp;&nbsp;&nbsp; Coordinates on the 3d cartesian space with shape==(3, ...)

## Examples

### Get the tiles touched by viewport

```python
import numpy as np

from py360tools.assets.projection_cmp import CMP

# Create a instance of projection
cmp = CMP(proj_res=f'600x400', tiling='6x4')

# Define the viewport position (in rads)
cmp.yaw_pitch_roll = np.deg2rad((70, 0, 0))

# Get the viewport tiles
viewport_tiles = cmp.get_vptiles()

print(viewport_tiles)  # ['3', '4', '5', '9', '10', '11', '12', '17']
```

### Draw a viewport over the projection

```python
import numpy as np
from PIL import Image
import py360tools.draw as draw

from py360tools import CMP, Viewport

# Open a projection frame
projection_array = np.array(Image.open('images/cmp1.png'))
height, width, _ = projection_array.shape

# Create a instance of projection
viewport = Viewport(resolution=f'220x180', fov='110x90')
viewport.projection = CMP(proj_res=f'{width}x{height}', tiling='1x1')

# Define the viewport position (in rads)
viewport.yaw_pitch_roll = np.deg2rad((70, 0, 0))

# Illuminates the viewport over the projection.
vp_mask = viewport.get_vp_mask(lum=255)
projection_array[vp_mask > 0] = (projection_array[vp_mask > 0] * 0.7 + 255 * 0.3).astype('uint8')

# change the pixel value of viewport border to blue
vp_border = viewport.get_vp_borders()
projection_array[vp_border > 0] = (0, 0, 255)

# Show the image
Image.fromarray(projection_array).show()
```

### Extract a viewport

```python
import numpy as np
from PIL import Image

from py360tools import CMP, Viewport

# Open a projection frame
frame_img = Image.open('images/cmp1.png')
frame_array = np.array(frame_img)
height, width, _ = frame_array.shape

# Create a instance of projection
viewport = Viewport(resolution=f'220x180', fov='110x90')
viewport.projection = CMP(proj_res=f'{width}x{height}', tiling='1x1')

# Define the viewport position (in rads)
yaw_pitch_roll = np.deg2rad((70, 0, 0))

# Get the viewport image
vp_image = viewport.extract_viewport(frame_array, yaw_pitch_roll)

# Show the image
Image.fromarray(vp_image).show()
```

## Other functions

- ```py360tools.util.splitx(string)```
- ```py360tools.util.unflatten_index(idx, shape)```
- ```py360tools.util.flatten_index(position, shape)```
- ```py360tools.util.mse2psnr(_mse, max_sample)```
- ```py360tools.util.make_tile_positions(tiling, proj_shape)```
- ```py360tools.util.iter_video(video_path, gray, dtype)```
- ```py360tools.util.check_ea(ea)```
- ```py360tools.util.get_borders_value(array, thickness)```
- ```py360tools.util.get_borders_coord_nm(position, shape)```
- ```py360tools.util.get_tile_borders(tile_id, tiling_shape, tile_shape)```
- ```py360tools.util.create_nm_coords(shape)```

## Depends

- `pillow`
- `numpy`
- `pandas`
- `opencv-python`

## Install
- `git clone https://github.com/henriquedgarcia/py360tools.git`
- `cd py360tools`
- `pip install -e .`

## Cite

Por favor, cite o trabalho abaixo caso você use esta ferramenta na sua pesquisa.

Referência:

```
Garcia, H. D., Carvalho, M. M. py360tool: Um Framework para Manipulação de Vídeo 360° com Ladrilhos. XXIV Workshop de Ferramentas e Aplicações (WFA 2025). Anais Estendidos do
XXXI Simpósio Brasileiro de Sistemas Multimídia e Web (WFA’2025). ISSN 2596-1683. Rio de Janeiro/RJ. (2025)
```

BibTex:

```bibtex
@inproceedings{Garcia2025,
   abstract = {The streaming of 360 ∘ videos is one of the most bandwidth-demanding virtual reality (VR) applications, as the video must be encoded in ultra-high resolution to ensure an im-mersive experience. To optimize its transmission, current approaches partition the spherical video into tiles, which are encoded at different bitrates and selectively delivered, based on the viewing direction of the user (viewport). The complexity of this architecture, which involves viewport prediction , tile selection, bit rate adaptation, and handling of parallel streaming, requires new tools to evaluate quality of experience (QoE) and quality of service (QoS), especially due to its interactive nature and low reproducibility. This work introduces py360tools, a Python library to handle tile-based 360 ∘ video streaming. The library automates key client-side tasks, such as spherical projection reconstruction, viewport extraction, and tile selection, facilitating the playback and simulation of streaming sessions. Furthermore, py360tools offers a flexible architecture, enabling efficient analysis of different projections and tiling strategies.},
   author = {Henrique D Garcia and Marcelo M Carvalho},
   city = {Rio de Janeiro},
   doi = {https://doi.org/10.48550/arXiv.2508.17428},
   issn = {2596-1683},
   booktitle = {Anais Estendidos do XXXI Simpósio Brasileiro de Sistemas Multimídia e Web (WFA’2025)},
   keywords = {Qualidade de Experiência,Qualidade de Serviço,Realidade Virtual,Streaming de Vídeo 360 ∘,Taxa de Bits Adaptativa},
   month = {11},
   pages = {1-4},
   publisher = {Brazilian Computer Society},
   title = {py360tool: Um Framework para Manipulação de Vídeo 360 ∘ com Ladrilhos},
   url = {https://www.meta.com/quest/quest-3/},
   year = {2025}
}

```

## py360tools Viewer

Um visualizador de demonstração da emulação do streming de vídeo 360º com ladrilhos. Disponível em: [Github - Py360tools Viewper](https://github.com/henriquedgarcia/py360tools_viewer) 

<img src="docs/img/py360tools_ui.png" alt="Descrição" width="300"/>

O py360tools Viewer recebe uma lista com os ladrilhos que serão vistos neste chunk e abre dois streams: um com a qualidade selecionada e outro de referência.
São calculados os MSEs dos ladrilhos e então calculada a média entre eles. Em seguida as projeções são extraídas o viewport que é usado par calcular o MSE.
Além do MSE médio dos ladrilhos e do MSE do viewport o visualizador ainda exibe o numero de ladrilhos selecionados e sua taxa de bits.

<img src="docs/img/aplicacao.png" alt="Descrição" width="300"/>

O JSON de entrada deve conter algumas informações básicas sobre as representações disponíveis, de forma semelhante ao DASH.

O segment_template deve conter obrigatoriamente as chaves:

- {tile} - Um identificador (espacial) do arquivo de ladrilho
- {quality} - Um identificador (qualidade) do fator de qualidade usado na codificação
- {chunk} - Um identificador (temporal) do chunk que será acessado.


<img src="docs/img/json_config.png" alt="Descrição" width="300"/>

O dataset do movimento de cabeça deve ser um pandas dataframe com multiindex serializado com HDF5 (chave única). O dataset deve conter uma amostra por quadro. Os campos do índice são "nome", "usuario" e "frame". A tabela possui três colunas ("yaw", "pitch", "roll") em radianos.

<img src="docs/img/hm_dataset.png" alt="Descrição" width="300"/>

Mais descrições em breve no Github do projeto.