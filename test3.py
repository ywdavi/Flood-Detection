import numpy as np

classes = ['Background', 'Building Flooded', 'Building Non-Flooded', 'Road Flooded',
           'Road Non-Flooded',  'Water', 'Tree', 'Vehicle',  'Pool', 'Grass']
RGBs = [[0.0, 0.0, 0.0],[255.0, 0.0, 0.0], [180.0, 120.0, 120.0], [160.0, 150.0, 20.0], [140.0, 140.0, 140.0],
        [61.0, 230.0, 250.0], [0.0, 82.0, 255.0],[255.0, 0.0, 245.0], [255.0, 235.0, 0.0], [4.0, 250.0, 7.0]]
normalized_palette = np.round(np.array(RGBs, dtype=np.float32) / 255 , 5)
normalized_palette_tuples = [tuple(color) for color in normalized_palette.tolist()]
classes_palette = dict(zip(normalized_palette_tuples, classes))
classes_to_index = {cls: idx for idx, cls in enumerate(classes)}
reverse_classes_palette = {v: k for k, v in classes_palette.items()}
print(reverse_classes_palette
      )