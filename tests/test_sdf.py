# dev tests for pytorch signed distance function
import unittest
import torch
import torch.nn.functional as F
import numpy as np
from torch.func import jacfwd, jacrev, vmap
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import gaussian_filter
import casadi as ca

from torch_traj_utils.scalar_field_interpolator import ScalarFieldInterpolator, SDF, SDF0, OcccupancyMapGenerator, OccupancyMap
from torch_traj_utils.scalar_field_interpolator_cas import ScalarFieldInterpolatorCas

class SDFTest(unittest.TestCase):

    def test_edt(self):
        # we need smooth (negative) SDT even inside obstacles
        occupancy = np.zeros((51,51), dtype=bool)
        occupancy[25:51,:] = 1
        plt.figure()
        plt.imshow(occupancy)
        plt.show()
        plt.figure()
        # these two factors account for outside and
        # inside obstacles
        sdf = edt(~occupancy) + -1.0*edt(occupancy)
        plt.imshow(sdf)
        plt.colorbar()
        plt.show()

    def test_room(self):
        sdf = SDF0(1, 1, 0.5, 0.5, .02)
        sdf.generate(0.14, 0.14)

    def test_room_new(self):
        omg = OcccupancyMapGenerator(1, 1, 0.5, 0.5, .02)
        occ_map = omg.generate_dd_test()
        sdf = SDF(occ_map, 0.14, 0.14)

    def test_sample_new(self):
        omg = OcccupancyMapGenerator(3, 2, -1.5, -1.0, .02)
        occ_map = omg.generate_dd_test()
        sdf = SDF(occ_map, 0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        dx = 0.05
        x = np.arange(sdf.ox, sdf.ox+sdf.x_size + dx, dx)
        y = np.arange(sdf.oy, sdf.oy+sdf.y_size + dx, dx)
        xx, yy = np.meshgrid(x, y)

        s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u_pts = np.zeros(s_pts.shape)
        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        c_np = c.detach().cpu().numpy()
        c_np = np.reshape(c_np, (41,61))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(c_np,
                   origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
                   aspect='auto',  # or 'equal'
                   cmap='viridis'
                   )
        plt.colorbar()
        plt.show()

    def test_sample_new_cas(self):
        # this should return the same plot as test_sample_new.
        omg = OcccupancyMapGenerator(3, 2, -1.5, -1.0, .02)
        occ_map = omg.generate_dd_test()
        sdf = SDF(occ_map, 0.14, 0.14)

        sfi = ScalarFieldInterpolatorCas(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        dx = 0.05
        x = np.arange(sdf.ox, sdf.ox+sdf.x_size + dx, dx)
        y = np.arange(sdf.oy, sdf.oy+sdf.y_size + dx, dx)
        xx, yy = np.meshgrid(x, y)

        s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u_pts = np.zeros(s_pts.shape)
        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        c = sfi.interpolator(s_pts, u_pts)
        c_np = np.reshape(c, (41,61))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(c_np,
                   origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
                   aspect='auto',  # or 'equal'
                   cmap='viridis'
                   )
        plt.colorbar()
        plt.show()

    def test_sample(self):
        sdf = SDF0(3, 2, -1.5, -1.0, .02)
        sdf.generate(0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        dx = 0.05
        x = np.arange(sdf.ox, sdf.ox+sdf.x_size + dx, dx)
        y = np.arange(sdf.oy, sdf.oy+sdf.y_size + dx, dx)
        xx, yy = np.meshgrid(x, y)

        s_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        u_pts = np.zeros(s_pts.shape)
        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        c_np = c.detach().cpu().numpy()
        c_np = np.reshape(c_np, (41,61))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(c_np,
                   origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],  # map array to coordinate bounds
                   aspect='auto',  # or 'equal'
                   cmap='viridis'
                   )
        plt.colorbar()
        plt.show()

    def test_jacobian_new(self):
        omg = OcccupancyMapGenerator(2, 2, -1, -1, .02)
        occ_map = omg.generate_dd_test()
        sdf = SDF(occ_map, 0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        s_pts = np.array([[-0.5, -0.5, 42], [0.0, -0.5, 43], [-0.5, 0.0, 44], [0.0, 0.0, 45]],dtype=np.float32)
        u_pts = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        jac_fun = jacrev(sfi.interpolator, argnums=(0, 1))
        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        A, B = vmap(jac_fun, in_dims=(0, 0))(S, U)  # A: (T,4), B: (T,1)

        d = c - torch.einsum('tij,tj->ti', A, S) - torch.einsum('tij,tj->ti', B, U)
        # back to numpy
        d_ret = d.detach().cpu().numpy()
        A_ret = A.detach().cpu().numpy()
        B_ret = B.detach().cpu().numpy()

        print(d_ret)
        print(A_ret)
        print(B_ret)

    def test_jacobian_new_cas(self):
        # this should return the same data as test_jacobian_new.
        # I won't be using this casadi functionality (it is for SCP),
        # but this was a fun exercise in seeing GPT5 translate the previous
        # test.
        omg = OcccupancyMapGenerator(2, 2, -1, -1, .02)
        occ_map = omg.generate_dd_test()
        sdf = SDF(occ_map, 0.14, 0.14)

        sfi = ScalarFieldInterpolatorCas(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        s_pts = np.array([[-0.5, -0.5, 42], [0.0, -0.5, 43], [-0.5, 0.0, 44], [0.0, 0.0, 45]],dtype=np.float32)
        u_pts = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

        T, n = s_pts.shape
        m = u_pts.shape[1]

        # ---- build symbolic variables ----
        S = ca.MX.sym("S", T, n)  # (T,n)
        U = ca.MX.sym("U", T, m)  # (T,m)

        c = sfi.interpolator(S, U)  # (T,1)

        T, n = s_pts.shape
        m = u_pts.shape[1]

        # ---- build a per-sample symbolic function (pure symbols) ----
        s = ca.MX.sym("s", n)  # (n,)
        u = ca.MX.sym("u", m)  # (m,)

        # reshape into batch-of-1 to reuse your interpolator which expects (N,n)
        S1 = ca.reshape(s, 1, n)  # (1,n)
        U1 = ca.reshape(u, 1, m)  # (1,m)

        c1 = sfi.interpolator(S1, U1)[0, 0]  # scalar MX

        A1 = ca.jacobian(c1, s)  # (1,n)
        B1 = ca.jacobian(c1, u)  # (1,m)

        per = ca.Function("per", [s, u], [c1, A1, B1])

        # ---- evaluate in a loop (unit-test friendly) ----
        c_list = []
        A_list = []
        B_list = []
        for t in range(T):
            ct, At, Bt = per(s_pts[t, :], u_pts[t, :])
            c_list.append(ct)
            A_list.append(At)
            B_list.append(Bt)

        c_dm = ca.vertcat(*c_list)  # (T,1)
        A_dm = ca.vertcat(*A_list)  # (T,n)
        B_dm = ca.vertcat(*B_list)  # (T,m)

        # ---- reproduce your d = c - A*S - B*U (row-wise dot products) ----
        S_dm = ca.DM(s_pts)  # (T,n)
        U_dm = ca.DM(u_pts)  # (T,m)

        d_dm = c_dm - ca.sum2(A_dm * S_dm) - ca.sum2(B_dm * U_dm)  # (T,1)

        # ---- back to numpy ----
        c_ret = np.array(c_dm)
        A_ret = np.array(A_dm)
        B_ret = np.array(B_dm)
        d_ret = np.array(d_dm)

        print(d_ret)
        print(A_ret)
        print(B_ret)

    def test_jacobian(self):
        sdf = SDF0(2, 2, -1, -1, .02)
        sdf.generate(0.14, 0.14)

        sfi = ScalarFieldInterpolator(sdf.sdf, sdf.ox, sdf.oy, sdf.res)

        s_pts = np.array([[-0.5, -0.5, 42], [0.0, -0.5, 43], [-0.5, 0.0, 44], [0.0, 0.0, 45]],dtype=np.float32)
        u_pts = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32)

        S = torch.from_numpy(s_pts)
        U = torch.from_numpy(u_pts)

        jac_fun = jacrev(sfi.interpolator, argnums=(0, 1))
        c = vmap(sfi.interpolator, in_dims=(0, 0))(S, U)  # (T,)
        A, B = vmap(jac_fun, in_dims=(0, 0))(S, U)  # A: (T,4), B: (T,1)

        d = c - torch.einsum('tij,tj->ti', A, S) - torch.einsum('tij,tj->ti', B, U)
        # back to numpy
        d_ret = d.detach().cpu().numpy()
        A_ret = A.detach().cpu().numpy()
        B_ret = B.detach().cpu().numpy()

        print(d_ret)
        print(A_ret)
        print(B_ret)


if __name__ == '__main__':
    unittest.main()
