# video360utils
Utilitários para transformações em vídeo 360. As projeções são subclasses de `lib.projectionbase.ProjBase`

```python
class lib.projectionbase.ProjBase(*, proj_res, fov, tiling = '1x1', vp_shape: = None):
```

**Parameters**
- `proj_res`: _str_ <br>
&nbsp;&nbsp;&nbsp;&nbsp; A string representing the projection resolution, e.g., '600x3000'.
- `fov`: _str_ <br>
&nbsp;&nbsp;&nbsp;&nbsp; A string representing the field of view (FOV), e.g., '100x90'.
- `tiling` (optional): _str_ <br>
&nbsp;&nbsp;&nbsp;&nbsp; A string representing the tiling, e.g., '1x1' or '3x2'.
- `vp_shape`:  _2-tuple_ or _ndarray_ <br>
&nbsp;&nbsp;&nbsp;&nbsp; The shape of the viewport, e.g., (300, 600) (height, width).

**Attributes**
- `yaw_pitch_roll`: 3-tuple or ndarray<br> 
&nbsp;&nbsp;&nbsp;&nbsp; A tuple or ndarray with three angles in radian. 

## Examples

### Get the tiles touched by viewport

```python
import numpy as np
from PIL import Image

from projections.cmp import CMP

# Create a instance of projection
cmp = CMP(proj_res=f'600x400', tiling='6x4', fov='110x90')

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

from projections.cmp import CMP

# Open a projection frame
frame_img = Image.open('images/cmp1.png')
frame_array = np.array(frame_img)
height, width, _ = frame_array.shape

# Create a instance of projection
cmp = CMP(proj_res=f'{width}x{height}', fov='110x90')

# Define the viewport position (in rads)
cmp.yaw_pitch_roll = np.deg2rad((70, 0, 0))

# Get the viewport and borders mask
vp_mask = cmp.draw_vp_mask(lum=255)
vp_border = cmp.draw_vp_borders()

# changes the pixel value to the weighted average of the pixel color and white
frame_array[vp_mask > 0] = (frame_array[vp_mask > 0] * 0.7 + 255 * 0.3).astype('uint8')

# change the pixel value to blue
frame_array[vp_border > 0] = (0, 0, 255)

# Show the image
Image.fromarray(frame_array).show()
```

### Extract a viewport

```python
import numpy as np
from PIL import Image

from projections.cmp import CMP

# Open a projection frame
frame_img = Image.open('images/cmp1.png')
frame_array = np.array(frame_img)
height, width, _ = frame_array.shape

# Create a instance of projection
cmp = CMP(proj_res=f'{width}x{height}', fov='110x90')

# Define the viewport position (in rads)
cmp.yaw_pitch_roll = np.deg2rad((70, 0, 0))

# Get the viewport image
vp_image = cmp.get_viewport_image(frame_array)

# Show the image
Image.fromarray(vp_image).show()
```

## Other functions
### ```lib.util.get_borders(*, coord_nm=None, shape=None, thickness=1)```

**Parameters**
- `coord_nm`: ndarray <br>
&nbsp;&nbsp;&nbsp;&nbsp; coord_nm must be a ndarray with shape==(chanel(C), height(N), width(M))
- `shape`: 2-tuple or ndarray<br>
&nbsp;&nbsp;&nbsp;&nbsp; A alternative shape, case coord_nm is a lists. (height, width) 
- `thickness`: int <br>
&nbsp;&nbsp;&nbsp;&nbsp; The border thickness in pixels

**Return**
- ndarray<br>
&nbsp;&nbsp;&nbsp;&nbsp; A ndarray with shape == (C, thickness*(2N+2M))
