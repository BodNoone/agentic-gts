import numpy as np

from agentic_gts.agent.ground import _GROUND_BACKGROUND, _projected_scatter
from agentic_gts.output.gs_render import Cam, rasterize_gs
from agentic_gts.tools.gs_io import GaussianData


def camera():
    return Cam(eye=np.array([0., 0., 4.]), target=np.zeros(3),
               up=np.array([0., 1., 0.]), fovy_deg=60, W=64, H=64)


def test_scatter_empty_background_is_muted_red():
    image = _projected_scatter(np.empty((0, 3)), camera(), 64, 64)
    assert np.allclose(image[0, 0], _GROUND_BACKGROUND, atol=1 / 255)


def test_background_forwarded_to_rasterizer_and_local_default_stays_black(monkeypatch):
    from agentic_gts.output import gs_render as renderer

    gs = GaussianData(means=np.zeros((1, 3), np.float32),
                      log_scales=np.zeros((1, 3), np.float32),
                      quats=np.array([[1., 0., 0., 0.]], np.float32),
                      raw_opacity=np.zeros(1, np.float32),
                      f_dc=np.zeros((1, 3), np.float32))
    calls = []

    def fake_backend(*args, **kwargs):
        calls.append(kwargs)
        return np.zeros((64, 64, 3), np.float32)

    monkeypatch.setattr(renderer, "_try_gsplat", fake_backend)
    rasterize_gs(gs, camera(), background=_GROUND_BACKGROUND)
    rasterize_gs(gs, camera())
    assert calls == [{"background": _GROUND_BACKGROUND}, {}]
