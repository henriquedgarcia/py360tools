# py360tools

Utilitários para transformações em vídeo 360. As projeções são subclasses de
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
cmp = CMP(proj_res=f'600x400', tiling='6x4', fov_res='110x90')

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

from py360tools.assets.projection_cmp import CMP

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

from py360tools.assets.projection_cmp import CMP

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

