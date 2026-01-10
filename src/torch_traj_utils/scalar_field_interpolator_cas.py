import numpy as np
import casadi as ca


class ScalarFieldInterpolatorCas:
    """
    CasADi version of your PyTorch ScalarFieldInterpolator.

    Conventions:
      - phi_np is (H, W) with access phi[y, x] (row=y, col=x), same as PyTorch.
      - xy is (N, 2) with columns [x_world, y_world]
      - returns are (N, 1)

    Differentiability:
      - piecewise smooth; non-smooth across pixel cell boundaries due to floor().
    """

    def __init__(self, phi_np: np.ndarray, origin_x: float, origin_y: float, resolution: float):
        phi_np = np.asarray(phi_np)
        assert phi_np.ndim == 2, "phi_np must be shape (H, W)"
        self.phi_np = phi_np
        self.H, self.W = phi_np.shape

        self.ox = float(origin_x)
        self.oy = float(origin_y)
        self.res = float(resolution)

        # Pixel index grids
        xg = np.arange(self.W, dtype=float)  # 0..W-1
        yg = np.arange(self.H, dtype=float)  # 0..H-1

        # CasADi interpolant expects data aligned with (x, y) indexing ("ij"), then flattened in Fortran order.
        # Our phi_np is (H, W) indexed as (y, x). Transpose -> (W, H) indexed as (x, y).
        data_xy = self.phi_np.T                      # (W, H)
        data_flat = data_xy.ravel(order="F")         # length W*H

        # 2D linear interpolant (bilinear in 2D)
        self._phi_interp = ca.interpolant("phi", "linear", [xg, yg], data_flat, {})

    @staticmethod
    def _clamp(v: ca.MX, lo: float, hi: float) -> ca.MX:
        return ca.fmin(ca.fmax(v, lo), hi)

    @staticmethod
    def _in_bounds(ix: ca.MX, iy: ca.MX, W: int, H: int) -> ca.MX:
        # Returns (N,1) mask in {0,1} as MX
        inx = ca.logic_and(ix >= 0, ix < W)
        iny = ca.logic_and(iy >= 0, iy < H)
        return ca.if_else(ca.logic_and(inx, iny), 1.0, 0.0)

    def _lut(self, x: ca.MX, y: ca.MX) -> ca.MX:
        """
        Lookup phi at (x, y) in pixel coordinates.
        x, y: (N,1) MX
        returns: (N,1) MX
        """
        pts = ca.vertcat(x.T, y.T)        # 2×N, each column is [x; y]
        return self._phi_interp(pts).T    # N×1

    def bilinear_sample(self, xy: ca.MX, padding_mode: str = "border") -> ca.MX:
        """
        xy: (N,2) MX in world coordinates
        returns: (N,1) MX
        """

        x = xy[:, 0]  # (N,1)
        y = xy[:, 1]  # (N,1)

        # Convert to pixel coordinates
        x_pix = (x - self.ox) / self.res
        y_pix = (y - self.oy) / self.res

        # Corner integer coordinates (still MX)
        x0 = ca.floor(x_pix)
        y0 = ca.floor(y_pix)
        x1 = x0 + 1
        y1 = y0 + 1

        # Weights use unclamped x0,y0 (matches your PyTorch version)
        wx = x_pix - x0
        wy = y_pix - y0

        if padding_mode == "border":
            # Clamp indices to border
            x0c = self._clamp(x0, 0.0, float(self.W - 1))
            x1c = self._clamp(x1, 0.0, float(self.W - 1))
            y0c = self._clamp(y0, 0.0, float(self.H - 1))
            y1c = self._clamp(y1, 0.0, float(self.H - 1))

            Ia = self._lut(x0c, y0c)
            Ib = self._lut(x1c, y0c)
            Ic = self._lut(x0c, y1c)
            Id = self._lut(x1c, y1c)

        elif padding_mode == "zeros":
            # For out-of-bounds corners, return 0 (per-corner), but still evaluate interpolant
            # at clamped coords and multiply by mask (safe_get behavior).
            x0c = self._clamp(x0, 0.0, float(self.W - 1))
            x1c = self._clamp(x1, 0.0, float(self.W - 1))
            y0c = self._clamp(y0, 0.0, float(self.H - 1))
            y1c = self._clamp(y1, 0.0, float(self.H - 1))

            ma = self._in_bounds(x0, y0, self.W, self.H)
            mb = self._in_bounds(x1, y0, self.W, self.H)
            mc = self._in_bounds(x0, y1, self.W, self.H)
            md = self._in_bounds(x1, y1, self.W, self.H)

            Ia = self._lut(x0c, y0c) * ma
            Ib = self._lut(x1c, y0c) * mb
            Ic = self._lut(x0c, y1c) * mc
            Id = self._lut(x1c, y1c) * md

        else:
            raise ValueError("padding_mode must be 'border' or 'zeros'")

        # Bilinear blend (N×1)
        vals = (
            Ia * (1 - wx) * (1 - wy) +
            Ib * wx       * (1 - wy) +
            Ic * (1 - wx) * wy +
            Id * wx       * wy
        )
        return vals

    def interpolator(self, s: ca.MX, u: ca.MX) -> ca.MX:
        """
        Batch analogue of your PyTorch usage.

        s: (N, n) MX states
        u: (N, m) MX controls (unused; kept for signature compatibility)
        returns: (N, 1) MX
        """
        xy = s[:, 0:2]  # (N,2) == PyTorch's s[:,:2]
        return self.bilinear_sample(xy, padding_mode="border")
