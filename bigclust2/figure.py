import cmap
import time

import pygfx as gfx
import numpy as np
import pylinalg as la

from functools import wraps
from rendercanvas.auto import RenderCanvas

from . import visuals


def update_figure(func):
    """Decorator to update figure."""

    @wraps(func)
    def inner(*args, **kwargs):
        val = func(*args, **kwargs)
        figure = args[0]

        # Any time we update the viewer, we should set it to stale
        figure._render_stale = True
        figure.canvas.request_draw()

        return val

    return inner


class BaseFigure:
    """Figure.

    Parameters
    ----------
    parent :    QWidget or None
                Parent widget, if any.
    max_fps :   int, optional
                Maximum frames per second to render.
    size :      tuple, optional
                Size of the viewer window.
    show :      bool, optional
                Whether to immediately show the viewer. Note that this has no
                effect in Jupyter. There you will have to call ``.show()`` manually
                on the last line of a cell for the viewer to appear.
    **kwargs
                Keyword arguments are passed through to ``WgpuCanvas``.

    """

    # Palette used for assigning colors to objects
    palette = "seaborn:tab10"

    def __init__(
        self,
        parent,
        max_fps=30,
        **kwargs,
    ):
        # Update some defaults as necessary
        defaults = {"max_fps": max_fps, "update_mode": "continuous"}
        if parent is not None:
            defaults["parent"] = parent
            defaults["size"] = (parent.size().width(), parent.size().height())
        defaults.update(kwargs)

        self.canvas = RenderCanvas(**defaults)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False)

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))

        # Stats
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        # Setup key events
        self.key_events = {}
        self.key_events["f"] = lambda: self.toggle_fps()

        def _keydown(event):
            """Handle key presses."""
            if not event.modifiers:
                if event.key in self.key_events:
                    self.key_events[event.key]()
            else:
                tup = (event.key, tuple(event.modifiers))
                if tup in self.key_events:
                    self.key_events[tup]()

        # Register events
        self.renderer.add_event_handler(_keydown, "key_down")

        # Finally, setting some variables
        self._animations = []
        self.render_trigger = "active_window" # "continuous", "reactive"

        # Set up the scene
        self.setup_scene()

    @property
    def size(self):
        """Return size of the canvas."""
        return self.canvas.get_logical_size()

    @size.setter
    @update_figure
    def size(self, size):
        """Set size of the canvas."""
        assert len(size) == 2
        self.canvas.set_logical_size(*size)

    @property
    def max_fps(self):
        """Maximum frames per second to render."""
        return self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps

    @max_fps.setter
    def max_fps(self, v):
        assert isinstance(v, int)
        self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps = v

    def toggle_fps(self):
        """Switch frames-per-second (FPS) display on and off."""
        self._show_fps = not self._show_fps

    def setup_scene(self):
        # Set up a default scene
        self.scene = gfx.Scene()
        # self.scene.add(gfx.AmbientLight(intensity=1))

        # Add the background (from BaseFigure) to the scene
        self.scene.add(gfx.Background(None, self._background))

        # Add a camera
        self.camera = gfx.OrthographicCamera()

        # Setup overlay
        self.overlay_camera = gfx.NDCCamera()
        self.overlay_scene = gfx.Scene()

        # Add a controller
        self.controller = gfx.PanZoomController(
            self.camera, register_events=self.renderer
        )

    @update_figure
    def add_animation(self, x):
        """Add animation function to the Viewer.

        Parameters
        ----------
        x :     callable
                Function to add to the animation loop.

        """
        if not callable(x):
            raise TypeError(f"Expected callable, got {type(x)}")

        self._animations.append(x)

    @update_figure
    def remove_animation(self, x):
        """Remove animation function from the Viewer.

        Parameters
        ----------
        x :     callable | int
                Either the function itself or its index
                in the list of animations.

        """
        if callable(x):
            self._animations.remove(x)
        elif isinstance(x, int):
            self._animations.pop(x)
        else:
            raise TypeError(f"Expected callable or index (int), got {type(x)}")

    def animate(self):
        """Run the render loop."""
        rm = self.render_trigger

        if rm == "active_window":
            # Note to self: we need to explore how to do this with different backends / Window managers
            if hasattr(self.canvas, "isActiveWindow"):
                if not self.canvas.isActiveWindow():
                    return
        elif rm == "reactive":
            # If the scene is not stale, we can skip rendering
            if not getattr(self, "_render_stale", False):
                return

        self.run_user_animations()

        # Now render the scene
        if self._show_fps:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
                self.renderer.render(
                    self.overlay_scene, self.overlay_camera, flush=False
                )
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera, flush=False)
            self.renderer.render(self.overlay_scene, self.overlay_camera)

        # Set stale to False
        self._render_stale = False

        self.canvas.request_draw()

    def force_single_render(self):
        """Force a single render of the scene."""
        self._render_stale = True
        self.renderer.render(self.scene, self.camera, flush=True)
        self.renderer.render(self.overlay_scene, self.overlay_camera)
        self.canvas.request_draw()

    def run_user_animations(self):
        """Run user-defined animations."""
        to_remove = []
        for i, func in enumerate(self._animations):
            try:
                func()
            except BaseException as e:
                print(f"Removing animation function {func} because of error: {e}")
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.remove_animation(i)

    @update_figure
    def show(self):
        """Show viewer."""
        # Start the animation loop
        self.canvas.request_draw(self.animate)

        # If this is an offscreen canvas, we don't need to show anything
        self.canvas.show()

    @update_figure
    def clear(self):
        """Clear scene of objects."""
        self.scene.clear()

    def is_visible_pos(self, pos, offset=0):
        """Test if positions are visible to camera."""
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 2
        assert pos.shape[1] == 2

        top_left = self.screen_to_world((0, 0))
        bottom_right = self.screen_to_world(self.size)

        is_visible = np.ones(len(pos), dtype=bool)

        is_visible[pos[:, 0] < top_left[0]] = False
        is_visible[pos[:, 1] > top_left[1]] = False
        is_visible[pos[:, 0] > bottom_right[0]] = False
        is_visible[pos[:, 1] < bottom_right[1]] = False

        return is_visible

    def screen_to_world(self, pos):
        """Translate screen position to world coordinates."""
        viewport = gfx.Viewport.from_viewport_or_renderer(self.renderer)
        if not viewport.is_inside(*pos):
            return None

        # Get position relative to viewport
        pos_rel = (
            pos[0] - viewport.rect[0],
            pos[1] - viewport.rect[1],
        )
        vs = viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += la.vec_transform(
            self.camera.world.position, self.camera.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self.camera.camera_matrix)

        return pos_world

    @update_figure
    def show_message(
        self, message, position="top-center", font_size=20, color=None, duration=None
    ):
        """Show message on canvas.

        Parameters
        ----------
        message :   str | None
                    Message to show. Set to `None` to remove the existing message.
        position :  "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center"
                    Position of the message on the canvas.
        font_size : int, optional
                    Font size of the message.
        color :     str | tuple, optional
                    Color of the message. If `None`, will use white.
        duration :  int, optional
                    Number of seconds after which to fade the message.

        """
        if message is None and hasattr(self, "_message_text"):
            if self._message_text.parent:
                self.overlay_scene.remove(self._message_text)
            del self._message_text
            return

        _positions = {
            "top-left": (-0.95, 0.95, 0),
            "top-right": (0.95, 0.95, 0),
            "top-center": (0, 0.95, 0),
            "bottom-left": (-0.95, -0.95, 0),
            "bottom-right": (0.95, -0.95, 0),
            "bottom-center": (0, -0.95, 0),
            "center": (0, 0, 0),
        }
        if position not in _positions:
            raise ValueError(f"Unknown position: {position}")

        if not hasattr(self, "_message_text"):
            self._message_text = visuals.text2gfx(
                message, color="white", font_size=font_size, screen_space=True
            )

        # Make sure the text is in the scene
        if self._message_text not in self.overlay_scene.children:
            self.overlay_scene.add(self._message_text)

        self._message_text.set_text(message)
        self._message_text.font_size = font_size
        self._message_text.anchor = position
        if color is not None:
            self._message_text.material.color = cmap.Color(color).rgba
        self._message_text.material.opacity = 1
        self._message_text.local.position = _positions[position]

        # When do we need to start fading out?
        if duration:
            self._fade_out_time = time.time() + duration

            def _fade_message():
                if not hasattr(self, "_message_text"):
                    self.remove_animation(_fade_message)
                else:
                    if time.time() > self._fade_out_time:
                        # This means the text will fade fade over 1/0.02 = 50 frames
                        self._message_text.material.opacity = max(
                            self._message_text.material.opacity - 0.02, 0
                        )

                    if self._message_text.material.opacity <= 0:
                        if self._message_text.parent:
                            self.overlay_scene.remove(self._message_text)
                        self.remove_animation(_fade_message)

            self.add_animation(_fade_message)
