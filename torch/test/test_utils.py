import unittest
import numpy as np
import os

class MyTestCase(unittest.TestCase):
    def test_store_load(self):
        s_s = np.random.rand(100, 4)
        u_s = np.random.rand(100, 1)
        np.savez("np_test.npz", s=s_s, u=u_s)

        stuff = np.load("np_test.npz")
        s_l = stuff['s']
        u_l = stuff['u']
        print('\n')
        print(np.max(s_l - s_s))
        print(np.max(u_l - u_s))

    def test_rand(self):
        low_vec = -np.array([0.1,2.0,0.5,12.0])
        high_vec = np.array([0.1, 2.0, 0.5, 12.0])
        one_pull = np.random.uniform(low_vec, high_vec, size=(10,4))
        print("\n")
        print(one_pull)

    def test_files(self):
        for f in os.listdir("trajectories_test"):
            print(f)

    def test_numpy_append(self):
        all_arr = []
        for i in range(0,3):
            arr = np.random.rand(8, 4)
            all_arr.append(arr)
        np_all_arr = np.concatenate(all_arr,0)
        print(np_all_arr)

    def test_numpy_bcast(self):
        norm = [1, 2, 3, 4]
        t = np.array([[1,1,1,1],[2,2,2,2]])
        print (t/norm)



if __name__ == '__main__':
    unittest.main()
