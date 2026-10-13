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


def test_legacy_gsplat_alpha_background_preserves_opaque_black(monkeypatch):
    import sys
    import types
    from agentic_gts.output.gs_render import _try_gsplat

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)

        def __getitem__(self, key):
            return Tensor(self.data[key])

        def squeeze(self, axis):
            return self

        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.data, axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.float32 = np.float32
    torch.tensor = lambda data, **kwargs: Tensor(data)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    # Legacy-compatible signature deliberately has no backgrounds keyword.
    def fake_rasterization(means, quats, scales, opac, rgb,
                           viewmats, Ks, width, height, render_mode):
        image = np.zeros((1, 1, 3, 3))
        image[0, 0, 2] = [0.2, 0.25, 0.3]
        alpha = np.array([[[[0.0], [1.0], [0.5]]]])
        return Tensor(image), Tensor(alpha), {}

    package = types.ModuleType("gsplat")
    rendering = types.ModuleType("gsplat.rendering")
    rendering.rasterization = fake_rasterization
    monkeypatch.setitem(sys.modules, "gsplat", package)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = _try_gsplat(np.zeros((1, 3)), np.array([[1., 0, 0, 0]]),
                        np.ones((1, 3)), np.ones(1), np.zeros((1, 3)),
                        np.eye(4), np.eye(3), 3, 1,
                        background=_GROUND_BACKGROUND)
    assert np.allclose(image[0, 0], _GROUND_BACKGROUND)
    assert np.allclose(image[0, 1], [0, 0, 0])
    assert np.allclose(image[0, 2],
                       np.array([0.2, 0.25, 0.3])
                       + 0.5 * np.array(_GROUND_BACKGROUND))
