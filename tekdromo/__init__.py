"""TEKDROMO - a Tektronix 4014 storage-tube face on a Jetson Nano.

Layering (each layer knows only about the ones below it):

    app        display loop, geometry cache, watchdog
    rig        controls -> regions -> expressions
    contour    the FDL contour generator
    field      the surface equation z(x,y)
    anatomy    measured shape + region constants + field primitives
    geometry / phosphor / framebuffer     transform, look, panel
"""
__version__ = "1.0"
