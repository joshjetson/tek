# face_2 — FDL v0.1 contour head

Implicit height field sliced into iso-contours, per the FDL spec.

  z(x,y) = skull + forehead + brow + nose + cheeks + lips + chin
           - eyes - philtrum - nostrils

  contour sweep : z = .95 -> -.25 step .05  (25 levels)
  regions       : 15 (skull, forehead, brow, orbit x2, nose bridge, nose tip,
                  nostrils x2, cheeks x2, philtrum, lips, chin, jaw, neck)
  output        : ~1,295 verts / ~1,457 edges
  build         : 4.2 s (one-off)

Why it differs from face_1: contours are LEVEL SETS of the real surface, so
they flow around every feature automatically. face_1 drew feature curves on
top of an undeformed mesh, which always read as a decal — most visibly on the
nose, where no amount of overlay produced a convincing ridge.

Run: tekface.py --model fdl
