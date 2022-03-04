import numpy as np
import pandas as pd
import torch
import sys
import os

def __init__():
    pass

class Basic_Setting():
    def __init__(self, filename):
        self.imp_numbers = 0
        self.filename = filename

    def size_parameter(self, **size):
        size.setdefault('nz', 1)
        size.setdefault('ns', 1)

        self.nx = size['nx']
        self.ny = size['ny']
        self.nz = size['nz']
        self.ns = size['ns']

        self.tsite = self.nx*self.ny*self.nz*self.ns

    def hopping_parameter(self, **hopping):
        hopping.setdefault('t1', 0)
        hopping.setdefault('t2', 0)
        hopping.setdefault('t3', 0)
        hopping.setdefault('t4', 0)
        hopping.setdefault('t5', 0)
        hopping.setdefault('t6', 0)
        hopping.setdefault('orbit_rotate', 0)
        
        self.t1 = hopping['t1']
        self.t2 = hopping['t2']
        self.t3 = hopping['t3']
        self.t4 = hopping['t4']
        self.t5 = hopping['t5']
        self.t6 = hopping['t6']
        self.orbit_rotate = hopping['orbit_rotate']

    def potential_parameter(self, mu = 0, u  = 0, jh = 0, v = 0):
        self.mu = mu
        self.u = u
        self.jh = jh
        self.u1 = u - 2.0*jh
        self.v = v

    def impurity_parameter(self, lattice, imp_numbers, imp_value, **imp_site):
        self.imp_numbers = imp_numbers
        self.imp_value = imp_value

        if imp_numbers == 1:
        
            imp_site.setdefault('csite', lattice.index[self.nx//2 - 1, self.ny//2 - 1, 0, 0])
            
            self.csite = imp_site['csite']
            self.filename = self.filename + 'imp' + str(imp_value)
            
        elif imp_numbers == 2:
        
            self.d = imp_site['d']
            imp_site.setdefault('csite1', lattice.index[self.nx//2 -  self.d//2 - 1      , self.ny//2 - 1, 0, 0])
            imp_site.setdefault('csite2', lattice.index[self.nx//2 + (self.d - self.d//2), self.ny//2 - 1, 0, 0])
            
            self.csite1 = imp_site['csite1']
            self.csite2 = imp_site['csite2']
            self.filename = self.filename + 'imp' + str(imp_value) + 'd' + str(self.d)

class Lattice:
    def __init__(self, setting):
        self.nx = setting.nx
        self.ny = setting.ny
        self.nz = setting.nz
        self.ns = setting.ns
        tsite = setting.tsite

        self.i0 = np.zeros(tsite, dtype = int)
        self.ix = np.zeros(tsite, dtype = int)
        self.iy = np.zeros(tsite, dtype = int)
        self.iz = np.zeros(tsite, dtype = int)
        self.isp = np.zeros(tsite, dtype = int)
        self.ipx = np.zeros(tsite, dtype = int)
        self.imx = np.zeros(tsite, dtype = int)
        self.ipy = np.zeros(tsite, dtype = int)
        self.imy = np.zeros(tsite, dtype = int)
        self.ipxpy = np.zeros(tsite, dtype = int)
        self.imxpy = np.zeros(tsite, dtype = int)
        self.ipxmy = np.zeros(tsite, dtype = int)
        self.imxmy = np.zeros(tsite, dtype = int)
        self.ipx2 = np.zeros(tsite, dtype = int)
        self.imx2 = np.zeros(tsite, dtype = int)
        self.ipy2 = np.zeros(tsite, dtype = int)
        self.imy2 = np.zeros(tsite, dtype = int)
        self.index = np.zeros((self.nx, self.ny, self.nz, self.ns), dtype = int)

    def lattice_square(self):
            nx = self.nx
            ny = self.ny
            nz = self.nz
            ns = self.ns

            i = 0
            for iy_ in range(ny):
                for ix_ in range(nx):
                    for isp_ in range(ns):
                        for iz_ in range(nz):
                            self.i0[i] = 1
                            self.ix[i] = ix_
                            self.iy[i] = iy_
                            self.iz[i] = iz_
                            self.isp[i] = isp_
                            self.index[ix_,iy_,iz_,isp_] = i
                            
                            px = ix_ + 1
                            py = iy_ + 1
                            if ix_ == nx - 1:
                                px = 0
                            if iy_ == ny - 1:
                                py = 0
                            self.ipx[i] = (iy_)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            self.ipy[i] = (py)*nx*nz*ns + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            mx = ix_ - 1
                            my = iy_ - 1
                            if ix_ == 0:
                                mx = nx - 1
                            if iy_ == 0:
                                my = ny - 1
                            self.imx[i] = (iy_)*nx*nz + (mx)*nz*ns + (isp_)*nz + iz_
                            self.imy[i] = (my)*nx*nz + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            self.ipxpy[i] = (py)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            self.imxpy[i] = (py)*nx*nz*ns + (mx)*nz*ns + (isp_)*nz + iz_
                            self.imxmy[i] = (my)*nx*nz*ns + (mx)*nz*ns + (isp_)*nz + iz_
                            self.ipxmy[i] = (my)*nx*nz*ns + (px)*nz*ns + (isp_)*nz + iz_
                            
                            px2 = ix_ + 2
                            py2 = iy_ + 2
                            if ix_ == nx - 1:
                                px2 = 1
                            if iy_ == ny - 1:
                                py2 = 1
                            if ix_ == nx - 2:
                                px2 = 0
                            if iy_ == ny - 2:
                                py2 = 0
                            self.ipx2[i] = (iy_)*nx*nz*ns + (px2)*nz*ns + (isp_)*nz + iz_
                            self.ipy2[i] = (py2)*nx*nz*ns + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            mx2 = ix_ - 2
                            my2 = iy_ - 2
                            if ix_ == 1:
                                mx2 = nx - 1
                            if iy_ == 1:
                                my2 = ny - 1
                            if ix_ == 0:
                                mx2 = nx - 2
                            if iy_ == 0:
                                my2 = ny - 2
                            self.imx2[i] = (iy_)*nx*nz + (mx2)*nz*ns + (isp_)*nz + iz_
                            self.imy2[i] = (my2)*nx*nz + (ix_)*nz*ns + (isp_)*nz + iz_
                            
                            if nx == 1:
                                self.ipx[i] = 0
                            if nx == 1:
                                self.imx[i] = 0
                            if ny == 1:
                                self.ipy[i] = 0
                            if ny == 1:
                                self.imy[i] = 0
                            
                            if nx == 1 or nx == 2:
                                self.ipx2[i] = 0
                            if nx == 1 or nx == 2:
                                self.imx2[i] = 0
                            if ny == 1 or ny == 2:
                                self.ipy2[i] = 0
                            if ny == 1 or ny == 2:
                                self.imy2[i] = 0
                            
                            if nx == 1 or ny == 1:
                                self.ipxpy[i] = 0
                            if nx == 1 or ny == 1:
                                self.imxpy[i] = 0
                            if nx == 1 or ny == 1:
                                self.imxmy[i] = 0
                            if nx == 1 or ny == 1:
                                self.ipxmy[i] = 0
                            i = i + 1
    
    def lattice_vacancy(self, i):
        self.i0[i] = 0

        self.ipx[self.imx[i]] = 0
        self.imx[self.ipx[i]] = 0
        self.ipy[self.imy[i]] = 0
        self.imy[self.ipy[i]] = 0

        self.ipxpy[self.imxmy[i]] = 0
        self.imxpy[self.ipxmy[i]] = 0
        self.ipxmy[self.imxpy[i]] = 0
        self.imxmy[self.ipxpy[i]] = 0

        self.ipx[i] = 0
        self.imx[i] = 0
        self.ipy[i] = 0
        self.imy[i] = 0

        self.ipxpy[i] = 0
        self.imxpy[i] = 0
        self.ipxmy[i] = 0
        self.imxmy[i] = 0

        self.ipx2[i] = 0
        self.imx2[i] = 0
        self.ipy2[i] = 0
        self.imy2[i] = 0

    def lattice_qnummap():
        pass

    def lattice_print():
        pass

    def lattice_vacancy_qnummap():
        pass

    def lattice_vacancy_qnummap_print():
        pass

class Read_File():
    def __init__(self, filename):
        self.filename = filename
        
    def s_wave(self, setting, lattice, **filename):
        filename.setdefault('filename', setting.filename)
        if os.path.isfile(filename['filename']):
            df = pd.read_excel(self.filename + '.xlsx')

            pair_array = np.array(df['pair_array'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.diag(pair_array)
        else:
            S_pz = 0.5 + 0.1*np.power(-1, lattice.ix)
            S_mz = 0.5 - 0.1*np.power(-1, lattice.ix)

            pair_array = np.linspace(0.1, 0.1, setting.tsite)
            pair = np.diag(pair_array)
        return pair, S_pz, S_mz

    def d_wave(self, setting, lattice, **filename):
        filename.setdefault('filename', setting.filename)
        if os.path.isfile(filename['filename']):
            df = pd.read_excel(self.filename + '.xlsx')

            pair_px = np.array(df['pair_px'])
            pair_py = np.array(df['pair_py'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.zeros((setting.tsite, setting.tsite))
            pair[np.arange(setting.tsite), lattice.ipx] = pair_px
            pair[np.arange(setting.tsite), lattice.ipy] = pair_py
            pair[lattice.ipx, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipx]
            pair[lattice.ipy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipy]
        else:
            S_pz = 0.5 + 0.1*np.power(-1, lattice.ix)
            S_mz = 0.5 - 0.1*np.power(-1, lattice.ix)

            pair_px = np.linspace(0.1, 0.1, setting.tsite)
            pair_py = np.linspace(0.1, 0.1, setting.tsite)
            
            pair[np.arange(setting.tsite), lattice.ipx] = pair_px
            pair[np.arange(setting.tsite), lattice.ipy] = pair_py
            pair[lattice.ipx, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipx]
            pair[lattice.ipy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipy]
        return pair, S_pz, S_mz

    def spm_wave(self, setting, lattice, **filename):
        filename.setdefault('filename', setting.filename)
        if os.path.isfile(filename['filename'] + '.xlsx'):
            df = pd.read_excel(self.filename + '.xlsx')

            pair_pxpy = np.array(df['pair_pxpy'])
            pair_pxmy = np.array(df['pair_pxmy'])
            S_pz = np.array(df['S_pz'])
            S_mz = np.array(df['S_mz'])

            pair = np.zeros((setting.tsite, setting.tsite))
            pair[np.arange(setting.tsite), lattice.ipxpy] = pair_pxpy
            pair[np.arange(setting.tsite), lattice.ipxmy] = pair_pxmy
            pair[lattice.ipxpy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipxpy]
            pair[lattice.ipxmy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipxmy]
        else:
            S_pz = 0.5 + 0.1*np.power(-1, lattice.ix)
            S_mz = 0.5 - 0.1*np.power(-1, lattice.ix)
            
            pair_pxpy = np.linspace(0.1, 0.1, setting.tsite)
            pair_pxmy = np.linspace(0.1, 0.1, setting.tsite)
            
            pair = np.zeros((setting.tsite, setting.tsite))
            pair[np.arange(setting.tsite), lattice.ipxpy] = pair_pxpy
            pair[np.arange(setting.tsite), lattice.ipxmy] = pair_pxmy
            pair[lattice.ipxpy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipxpy]
            pair[lattice.ipxmy, np.arange(setting.tsite)] = pair[np.arange(setting.tsite), lattice.ipxmy]
        return pair, S_pz, S_mz

class Selfconsistent:
    def __init__(self, **selfconsistentParameter):

        selfconsistentParameter.setdefault('converge_spin', False)
        selfconsistentParameter.setdefault('converge_pair', False)
        
        selfconsistentParameter.setdefault('alpha_spin', 0.5)
        selfconsistentParameter.setdefault('alpha_pair', 0.5)
        
        selfconsistentParameter.setdefault('tol_spin', 1.0E-4)
        selfconsistentParameter.setdefault('tol_pair', 1.0E-4)
        
        self.doping = 0.0

        self.converge_spin = selfconsistentParameter['converge_spin']
        self.alpha_spin = selfconsistentParameter['alpha_spin']
        self.tol_spin = selfconsistentParameter['tol_spin']

        self.converge_pair = selfconsistentParameter['converge_pair']
        self.alpha_pair = selfconsistentParameter['alpha_pair']
        self.tol_pair = selfconsistentParameter['tol_pair']

        self.converge_total = self.converge_spin and self.converge_pair

        selfconsistentParameter.setdefault('T', 0.001)
        self.T = selfconsistentParameter['T']
    
    def get_doping_value(self, setting, eig_vec, eig_val):
        ff = 0.5*(1-np.tanh(0.5*eig_val/self.T))
        eig_vec_up = np.split(eig_vec, 2)[0]
        eig_vec_dn = np.split(eig_vec, 2)[1]

        doping = np.average((np.abs(eig_vec_up)**2).dot(ff) + (np.abs(eig_vec_dn)**2).dot(1.0-ff))
        self.doping = doping*setting.nz*setting.ns
        return self.doping
        
    # electrons
    def electrons(self, setting, eig_vec, eig_val, spin_up, spin_dn):
        ff = 0.5*(1-np.tanh(0.5*eig_val/self.T))
        eig_vec_up = np.split(eig_vec, 2)[0]
        eig_vec_dn = np.split(eig_vec, 2)[1]

        new_spin_up = (np.abs(eig_vec_up)**2).dot(ff)
        new_spin_dn = (np.abs(eig_vec_dn)**2).dot((1.0 - ff))
        # Also can use np.einsum to obtain the same result.
        # new_spin_up = np.einsum('ij, j->i', np.abs(eig_vec_up)**2, ff)
        # new_spin_dn = np.einsum('ij, j->i', np.abs(eig_vec_up)**2, (1.0 - ff))
            
        doping = np.average(new_spin_up + new_spin_dn)
        old_doping = np.average(spin_up + spin_dn)
        if (np.abs(old_doping - doping) > self.tol_spin):
            self.converge_spin = False
            spin_up = spin_up*(1.0 - self.alpha_spin) + new_spin_up*self.alpha_spin
            spin_dn = spin_dn*(1.0 - self.alpha_spin) + new_spin_dn*self.alpha_spin
        else:
            self.converge_spin = True
        self.doping = doping*setting.nz*setting.ns
        return spin_up, spin_dn
   
    def s_wave_pair(self, setting, lattice, eig_vec, eig_val, pair):
        vec_up = eig_vec[0:setting.tsite, :]
        vec_down = eig_vec[setting.tsite:2*setting.tsite, :]
        
        new_pair = np.multiply(vec_up, vec_down).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5
        
        pair_array = np.diag(pair)

        s_wave = np.average(pair)
        new_s_wave = np.average(new_pair)
        if (np.abs(s_wave - new_s_wave) > self.tol_pair):
            self.converge_pair = False
            new_pair = pair_array*(1.0 - self.alpha_pair) + new_pair*self.alpha_pair

            pair = np.diag(pair_array)
        else:
            self.converge_pair = True
        return pair, new_pair
    
    # d_wave_pair
    def d_wave_pair(self, setting, lattice, eig_vec, eig_val, pair):
        vec_up = eig_vec[0:setting.tsite, :]
        vec_down_px = eig_vec[lattice.ipx + setting.tsite, :]
        vec_down_py = eig_vec[lattice.ipy + setting.tsite, :]
        
        new_pair_px = np.multiply(vec_up, vec_down_px).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5
        new_pair_py = np.multiply(vec_up, vec_down_py).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5

        pair_px = pair[np.arange(pair.shape[0]), lattice.ipx]
        pair_py = pair[np.arange(pair.shape[0]), lattice.ipy]

        d_wave = np.average(pair_px + pair_py)
        new_d_wave = np.average(new_pair_px + new_pair_py)
        if (np.abs(d_wave - new_d_wave) > self.tol_pair):
            self.converge_pair = False
            new_pair_px = pair_px*(1.0 - self.alpha_pair) + new_pair_px*self.alpha_pair
            new_pair_py = pair_py*(1.0 - self.alpha_pair) + new_pair_py*self.alpha_pair

            pair[np.arange(pair.shape[0]), lattice.ipx] = new_pair_px
            pair[np.arange(pair.shape[0]), lattice.ipy] = new_pair_py
            pair[lattice.ipx, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipx]
            pair[lattice.ipy, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipy]
        else:
            self.converge_pair = True
        return pair, new_pair_px, new_pair_py

    # spm_wave_pair
    def spm_wave_pair(self, setting, lattice, eig_vec, eig_val, pair):
            vec_up = eig_vec[0:setting.tsite, :]
            vec_down_pxpy = eig_vec[lattice.ipxpy + setting.tsite, :]
            vec_down_pxmy = eig_vec[lattice.ipxmy + setting.tsite, :]
            
            new_pair_pxpy = np.multiply(vec_up, vec_down_pxpy).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5
            new_pair_pxmy = np.multiply(vec_up, vec_down_pxmy).dot(np.tanh(0.5*eig_val/self.T))*setting.v*0.5

            pair_pxpy = pair[np.arange(pair.shape[0]), lattice.ipxpy]
            pair_pxmy = pair[np.arange(pair.shape[0]), lattice.ipxmy]

            spm_wave = np.average(pair_pxpy + pair_pxmy)
            new_spm_wave = np.average(new_pair_pxpy + new_pair_pxmy)
            if (np.abs(spm_wave - new_spm_wave) > self.tol_pair):
                self.converge_pair = False
                new_pair_pxpy = pair_pxpy*(1.0 - self.alpha_pair) + new_pair_pxpy*self.alpha_pair
                new_pair_pxmy = pair_pxmy*(1.0 - self.alpha_pair) + new_pair_pxmy*self.alpha_pair

                pair[np.arange(pair.shape[0]), lattice.ipxpy] = new_pair_pxpy
                pair[np.arange(pair.shape[0]), lattice.ipxmy] = new_pair_pxmy
                pair[lattice.ipxpy, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipxpy]
                pair[lattice.ipxmy, np.arange(pair.shape[1])] = pair[np.arange(pair.shape[0]), lattice.ipxmy]
            else:
                self.converge_pair = True
            return pair, new_pair_pxpy, new_pair_pxmy

class Hamiltonian():
    def selfconsistent(setting, lattice, pair, S_pz, S_mz):
        tsite = setting.tsite
        t1 = setting.t1
        t2 = setting.t2
        t3 = setting.t3
        t4 = setting.t4
        t5 = setting.t5
        t6 = setting.t6

        mu = setting.mu
        u  = setting.u
        jh = setting.jh
        u1 = setting.u1

        H = np.zeros((tsite*2, tsite*2), dtype = np.float64)
        iz = np.array([1, -1]*(tsite//2))
        i = np.arange(tsite)
        
        H[np.arange(tsite)      , np.arange(tsite)      ] =   -mu + u*S_mz + u1*S_mz[i+iz] + (u1-jh)*S_pz[i+iz]
        H[np.arange(tsite)+tsite, np.arange(tsite)+tsite] = -(-mu + u*S_pz + u1*S_pz[i+iz] + (u1-jh)*S_mz[i+iz])
        
        if setting.imp_numbers == 1:
            imp_site = np.array([setting.csite, setting.csite + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        elif setting.imp_numbers == 2:
            imp_site = np.array([setting.csite1, setting.csite1 + 1, setting.csite2, setting.csite2 + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        
        H[i, lattice.ipx] = -t1
        H[i, lattice.imx] = -t1
        H[i, lattice.ipy] = -t1
        H[i, lattice.imy] = -t1
        
        t2p = np.full(tsite, t2)
        t3p = np.full(tsite, t3)
        if setting.orbit_rotate == 90:
            logical_array = np.logical_or(np.logical_and((lattice.ix + lattice.iy) % 2 == 0, iz == 1), np.logical_and((lattice.ix + lattice.iy) % 2 == 1, iz == -1))

            t2p[np.where(logical_array == False)[0]] = t3
            t3p[np.where(logical_array == False)[0]] = t2
        
        H[i, lattice.ipxpy] = -t2p
        H[i, lattice.imxpy] = -t3p
        H[i, lattice.imxmy] = -t2p
        H[i, lattice.ipxmy] = -t3p
        
        
        H[i, lattice.ipxpy+iz] = -t4
        H[i, lattice.imxpy+iz] = -t4
        H[i, lattice.imxmy+iz] = -t4
        H[i, lattice.ipxmy+iz] = -t4
        
        H[i, lattice.ipx+iz] = -t5
        H[i, lattice.ipy+iz] = -t5
        H[i, lattice.imx+iz] = -t5
        H[i, lattice.imy+iz] = -t5

        H[i, lattice.ipx2] = -t6
        H[i, lattice.ipy2] = -t6
        H[i, lattice.imx2] = -t6
        H[i, lattice.imy2] = -t6

        H[i+tsite, lattice.ipx+tsite] = -np.conj(H[i, lattice.ipx])
        H[i+tsite, lattice.imx+tsite] = -np.conj(H[i, lattice.imx])
        H[i+tsite, lattice.ipy+tsite] = -np.conj(H[i, lattice.ipy])
        H[i+tsite, lattice.imy+tsite] = -np.conj(H[i, lattice.imy])

        H[i+tsite, lattice.ipxpy+tsite] = -np.conj(H[i, lattice.ipxpy])
        H[i+tsite, lattice.imxpy+tsite] = -np.conj(H[i, lattice.imxpy])
        H[i+tsite, lattice.imxmy+tsite] = -np.conj(H[i, lattice.imxmy])
        H[i+tsite, lattice.ipxmy+tsite] = -np.conj(H[i, lattice.ipxmy])

        H[i+tsite, lattice.ipxpy+tsite+iz] = -np.conj(H[i, lattice.ipxpy+iz])
        H[i+tsite, lattice.imxpy+tsite+iz] = -np.conj(H[i, lattice.imxpy+iz])
        H[i+tsite, lattice.imxmy+tsite+iz] = -np.conj(H[i, lattice.imxmy+iz])
        H[i+tsite, lattice.ipxmy+tsite+iz] = -np.conj(H[i, lattice.ipxmy+iz])

        H[i+tsite, lattice.ipx+tsite+iz] = -np.conj(H[i, lattice.ipx+iz])
        H[i+tsite, lattice.ipy+tsite+iz] = -np.conj(H[i, lattice.ipy+iz])
        H[i+tsite, lattice.imx+tsite+iz] = -np.conj(H[i, lattice.imx+iz])
        H[i+tsite, lattice.imy+tsite+iz] = -np.conj(H[i, lattice.imy+iz])

        H[i+tsite, lattice.ipx2+tsite] = -np.conj(H[i, lattice.ipx2])
        H[i+tsite, lattice.ipy2+tsite] = -np.conj(H[i, lattice.ipy2])
        H[i+tsite, lattice.imx2+tsite] = -np.conj(H[i, lattice.imx2])
        H[i+tsite, lattice.imy2+tsite] = -np.conj(H[i, lattice.imy2])

        H[i, lattice.ipxpy+tsite] = pair[i, lattice.ipxpy]
        H[i, lattice.imxpy+tsite] = pair[i, lattice.imxpy]
        H[i, lattice.ipxmy+tsite] = pair[i, lattice.ipxmy]
        H[i, lattice.imxmy+tsite] = pair[i, lattice.imxmy]

        H[i+tsite, lattice.ipxpy] = pair[lattice.ipxpy, i]
        H[i+tsite, lattice.imxpy] = pair[lattice.imxpy, i]
        H[i+tsite, lattice.ipxmy] = pair[lattice.ipxmy, i]
        H[i+tsite, lattice.imxmy] = pair[lattice.imxmy, i]
        return H

    def supercell(setting, lattice, pair, S_pz, S_mz, pfx, pfy):
        tsite = setting.tsite
        t1 = setting.t1
        t2 = setting.t2
        t3 = setting.t3
        t4 = setting.t4
        t5 = setting.t5
        t6 = setting.t6

        mu = setting.mu
        u  = setting.u
        jh = setting.jh
        u1 = setting.u1

        H = np.zeros((tsite*2, tsite*2), dtype = np.complex64)
        iz = np.array([1, -1]*(tsite//2))
        i = np.arange(tsite)
        
        H[np.arange(tsite)      , np.arange(tsite)      ] =   -mu + u*S_mz + u1*S_mz[i+iz] + (u1-jh)*S_pz[i+iz]
        H[np.arange(tsite)+tsite, np.arange(tsite)+tsite] = -(-mu + u*S_pz + u1*S_pz[i+iz] + (u1-jh)*S_mz[i+iz])
        
        if setting.imp_numbers == 1:
            imp_site = np.array([setting.csite, setting.csite + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        elif setting.imp_numbers == 2:
            imp_site = np.array([setting.csite1, setting.csite1 + 1, setting.csite2, setting.csite2 + 1])
            H[imp_site,       imp_site]       = H[imp_site,       imp_site]       + setting.imp_value
            H[imp_site+tsite, imp_site+tsite] = H[imp_site+tsite, imp_site+tsite] - setting.imp_value
        
        H[i, lattice.ipx] = -t1
        H[i, lattice.imx] = -t1
        H[i, lattice.ipy] = -t1
        H[i, lattice.imy] = -t1
        
        t2p = np.full(tsite, t2)
        t3p = np.full(tsite, t3)
        
        if setting.orbit_rotate == 90:
            logical_array = np.logical_or(np.logical_and((lattice.ix + lattice.iy) % 2 == 0, iz == 1), np.logical_and((lattice.ix + lattice.iy) % 2 == 1, iz == -1))

            t2p[np.where(logical_array == False)[0]] = t3
            t3p[np.where(logical_array == False)[0]] = t2
        
        H[i, lattice.ipxpy] = -t2p
        H[i, lattice.imxpy] = -t3p
        H[i, lattice.imxmy] = -t2p
        H[i, lattice.ipxmy] = -t3p
        
        H[i, lattice.ipxpy+iz] = -t4
        H[i, lattice.imxpy+iz] = -t4
        H[i, lattice.imxmy+iz] = -t4
        H[i, lattice.ipxmy+iz] = -t4
        
        H[i, lattice.ipx+iz] = -t5
        H[i, lattice.ipy+iz] = -t5
        H[i, lattice.imx+iz] = -t5
        H[i, lattice.imy+iz] = -t5

        H[i, lattice.ipx2] = -t6
        H[i, lattice.ipy2] = -t6
        H[i, lattice.imx2] = -t6
        H[i, lattice.imy2] = -t6
        
        H[i+tsite, lattice.ipx+tsite] = -np.conj(H[i, lattice.ipx])*pfx
        H[i+tsite, lattice.imx+tsite] = -np.conj(H[i, lattice.imx])*np.conj(pfx)
        H[i+tsite, lattice.ipy+tsite] = -np.conj(H[i, lattice.ipy])*pfy
        H[i+tsite, lattice.imy+tsite] = -np.conj(H[i, lattice.imy])*np.conj(pfy)

        H[i+tsite, lattice.ipxpy+tsite] = -np.conj(H[i, lattice.ipxpy])*pfx*pfy
        H[i+tsite, lattice.imxpy+tsite] = -np.conj(H[i, lattice.imxpy])*np.conj(pfx)*pfy
        H[i+tsite, lattice.imxmy+tsite] = -np.conj(H[i, lattice.imxmy])*np.conj(pfx)*np.conj(pfy)
        H[i+tsite, lattice.ipxmy+tsite] = -np.conj(H[i, lattice.ipxmy])*pfx*np.conj(pfy)

        H[i+tsite, lattice.ipxpy+tsite+iz] = -np.conj(H[i, lattice.ipxpy+iz])*pfx*pfy
        H[i+tsite, lattice.imxpy+tsite+iz] = -np.conj(H[i, lattice.imxpy+iz])*np.conj(pfx)*pfy
        H[i+tsite, lattice.imxmy+tsite+iz] = -np.conj(H[i, lattice.imxmy+iz])*np.conj(pfx)*np.conj(pfy)
        H[i+tsite, lattice.ipxmy+tsite+iz] = -np.conj(H[i, lattice.ipxmy+iz])*pfx*np.conj(pfy)

        H[i+tsite, lattice.ipx+tsite+iz] = -np.conj(H[i, lattice.ipx+iz])*pfx
        H[i+tsite, lattice.ipy+tsite+iz] = -np.conj(H[i, lattice.ipy+iz])*pfy
        H[i+tsite, lattice.imx+tsite+iz] = -np.conj(H[i, lattice.imx+iz])*np.conj(pfx)
        H[i+tsite, lattice.imy+tsite+iz] = -np.conj(H[i, lattice.imy+iz])*np.conj(pfy)
        
        H[i+tsite, lattice.ipx2+tsite] = -np.conj(H[i, lattice.ipx2])*pfx*pfx
        H[i+tsite, lattice.ipy2+tsite] = -np.conj(H[i, lattice.ipy2])*pfy*pfy
        H[i+tsite, lattice.imx2+tsite] = -np.conj(H[i, lattice.imx2])*np.conj(pfx)*np.conj(pfx)
        H[i+tsite, lattice.imy2+tsite] = -np.conj(H[i, lattice.imy2])*np.conj(pfy)*np.conj(pfy)

        H[i, lattice.ipx] = H[i, lattice.ipx]*pfx
        H[i, lattice.imx] = H[i, lattice.imx]*np.conj(pfx)
        H[i, lattice.ipy] = H[i, lattice.ipy]*pfy
        H[i, lattice.imy] = H[i, lattice.imy]*np.conj(pfy)

        H[i, lattice.ipxpy] = H[i, lattice.ipxpy]*pfx*pfy
        H[i, lattice.imxpy] = H[i, lattice.imxpy]*np.conj(pfx)*pfy
        H[i, lattice.imxmy] = H[i, lattice.imxmy]*np.conj(pfx)*np.conj(pfy)
        H[i, lattice.ipxmy] = H[i, lattice.ipxmy]*pfx*np.conj(pfy)

        H[i, lattice.ipxpy+iz] = H[i, lattice.ipxpy+iz]*pfx*pfy
        H[i, lattice.imxpy+iz] = H[i, lattice.imxpy+iz]*np.conj(pfx)*pfy
        H[i, lattice.imxmy+iz] = H[i, lattice.imxmy+iz]*np.conj(pfx)*np.conj(pfy)
        H[i, lattice.ipxmy+iz] = H[i, lattice.ipxmy+iz]*pfx*np.conj(pfy)

        H[i, lattice.ipx+iz] = H[i, lattice.ipx+iz]*pfx
        H[i, lattice.ipy+iz] = H[i, lattice.ipy+iz]*pfy
        H[i, lattice.imx+iz] = H[i, lattice.imx+iz]*np.conj(pfx)
        H[i, lattice.imy+iz] = H[i, lattice.imy+iz]*np.conj(pfy)

        H[i, lattice.ipx2] = H[i, lattice.ipx2]*pfx*pfx
        H[i, lattice.ipy2] = H[i, lattice.ipy2]*pfy*pfy
        H[i, lattice.imx2] = H[i, lattice.imx2]*np.conj(pfx)*np.conj(pfx)
        H[i, lattice.imy2] = H[i, lattice.imy2]*np.conj(pfy)*np.conj(pfy)

        H[i, lattice.ipxpy+tsite] = pair[i, lattice.ipxpy]*pfx*pfy
        H[i, lattice.imxpy+tsite] = pair[i, lattice.imxpy]*np.conj(pfx)*pfy
        H[i, lattice.ipxmy+tsite] = pair[i, lattice.ipxmy]*pfx*np.conj(pfy)
        H[i, lattice.imxmy+tsite] = pair[i, lattice.imxmy]*np.conj(pfx)*np.conj(pfy)
        
        H[i+tsite, lattice.ipxpy] = pair[lattice.ipxpy, i]*pfx*pfy
        H[i+tsite, lattice.imxpy] = pair[lattice.imxpy, i]*np.conj(pfx)*pfy
        H[i+tsite, lattice.ipxmy] = pair[lattice.ipxmy, i]*pfx*np.conj(pfy)
        H[i+tsite, lattice.imxmy] = pair[lattice.imxmy, i]*np.conj(pfx)*np.conj(pfy)
        return H