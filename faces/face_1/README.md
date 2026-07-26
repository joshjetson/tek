# face_1 — measured-reconstruction head

Built by measuring the free3d reference against a normalised grid
(unit = crown-to-chin height, origin = chin on the midline).

  silhouette ratio vs reference : 0.996
  landmark mean error           : 3.8%  (11 landmarks)
  eye aperture                  : 0.155H x 0.051H, canthal tilt +0.022H
  mouth                         : corners +-0.115H, bow peaks y=0.133,
                                  lower lip 1.5x upper
  edges                         : ~5,977 static + ~190 animated mouth
  live framerate                : 29 fps at 1024x600

Method: dense lat/long base ovoid, ~14 anatomical displacements, tangential
warp field, plus feature outline curves. Talking mouth regenerated per frame
from (openness, rounding).

Entry point: tekface.py --model anat
