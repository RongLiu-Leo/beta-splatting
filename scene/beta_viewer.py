import nerfview
import viser
from typing import Literal
from typing import Callable


class BetaViewer(nerfview.Viewer):
    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        super().__init__(server, render_fn, mode=mode)
        server.gui.set_panel_label("Beta Splatting Viewer")

    def _populate_rendering_tab(self):
        with self._rendering_folder:
            with self.server.gui.add_folder("Geometry Complexity Control"):
                self.gui_multi_slider = self.server.gui.add_multi_slider(
                    "b Range",
                    min=-1,
                    max=1,
                    step=0.01,
                    initial_value=(-1, 1),
                )
                self.gui_multi_slider.on_update(self.rerender)
            with self.server.gui.add_folder("Render Mode"):
                self.gui_dropdown = self.server.gui.add_dropdown(
                    "Mode",
                    ["RGB", "Diffuse", "Specular", "Depth"],
                    initial_value="RGB",
                )
                self.gui_dropdown.on_update(self.rerender)
        super()._populate_rendering_tab()
