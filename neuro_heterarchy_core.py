#!/usr/bin/env python3
"""
🧠 NEURO-HETERARCHY CORE 7.5 (STRICT HILBERT & SUB-BIN CONTINUOUS STREAMING)
- Аналитический сигнал Гильберта Z(t) = x(t) + i*H(x) (знак iPLV сохранен на 100%).
- Непрерывный спектральный центр тяжести Теты (без дискретных скачков 126/176/192).
- Матрица 32x120 нарезается фазой Теты в реальном времени.
"""

import os
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import time
import math
import ctypes
import numpy as np
from dataclasses import dataclass
import torch
from pylsl import StreamInlet, resolve_byprop

FS = 250.0
CHUNK_SIZE = 8
NUM_CHANNELS = 16
NUM_DEVICES = 4
NUM_FREQS = 32
NUM_PAIRS = 120

COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.43, -10.14,
   -10.14, -7.42, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
     2.71,  7.42,  4.77, 10.15, 10.14,  4.76,  7.43,   2.72,
    -2.73, -7.42, -4.77,-10.14,-10.15, -4.77, -7.43,  -2.72
], dtype=np.float32)

I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)
DX_PAIR = COORDS_X[J_IDX] - COORDS_X[I_IDX]
DY_PAIR = COORDS_Y[J_IDX] - COORDS_Y[I_IDX]
TQ_MULT = (COORDS_X[I_IDX] * DY_PAIR - COORDS_Y[I_IDX] * DX_PAIR) / 100.0

@dataclass
class UniversalGamepadAxes:
    lx: float
    ly: float
    rx: float
    ry: float

@dataclass
class NodeState:
    name: str
    device_id: int
    vx: float
    vy: float
    tq: float
    thrust: float
    traj_32: np.ndarray       
    iplv_32: np.ndarray       
    
    @property
    def gamepad_axes(self) -> UniversalGamepadAxes:
        base_x, base_y = self.traj_32[0, 0], self.traj_32[0, 1]
        end_x = self.traj_32[-1, 0] - base_x
        end_y = self.traj_32[-1, 1] - base_y
        d_len = math.hypot(end_x, end_y) + 1e-6

        lx = float(end_x / d_len if d_len > 1.0 else end_x)
        ly = float(end_y / d_len if d_len > 1.0 else end_y)

        sagitta_sum = 0.0
        for k in range(1, NUM_FREQS - 1):
            px_k = self.traj_32[k, 0] - base_x
            py_k = self.traj_32[k, 1] - base_y
            sagitta_sum += (end_x * py_k - end_y * px_k)
        rx = float(np.clip(sagitta_sum / (d_len * 16.0), -1.0, 1.0))

        mid_idx = NUM_FREQS // 2
        len_past = math.hypot(self.traj_32[mid_idx, 0] - base_x, self.traj_32[mid_idx, 1] - base_y)
        len_future = math.hypot(end_x - (self.traj_32[mid_idx, 0] - base_x), end_y - (self.traj_32[mid_idx, 1] - base_y))
        ry = float((len_future - len_past) / (len_future + len_past + 1e-6))
        
        return UniversalGamepadAxes(lx, -ly, rx, ry)

@dataclass
class MultimodalFrame:
    fcz_macro: NodeState
    pz_spatial: NodeState
    oz_sensory: NodeState
    cz_motor: NodeState
    theta_freq: float
    theta_sync: float
    is_real: bool
    num_live: int

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem

    def run(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CORE ENGINE] Strict Hilbert Streaming PAC Decoder on {DEVICE}...")

        I_GPU = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
        J_GPU = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)
        DX_GPU = torch.from_numpy(DX_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        DY_GPU = torch.from_numpy(DY_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        TQ_GPU = torch.from_numpy(TQ_MULT).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)

        BUF_LEN = 256
        eeg_ring = torch.zeros((NUM_DEVICES, NUM_CHANNELS, BUF_LEN), device=DEVICE, dtype=torch.float32)
        
        # Сетка частот полного комплексного БПФ (размер 256)
        freqs = torch.fft.fftfreq(BUF_LEN, d=1.0/FS).to(DEVICE)
        
        # Маска строго положительных частот Теты (размер 256)
        theta_mask = (freqs >= 3.8) & (freqs <= 7.8)
        theta_freqs_slice = freqs[theta_mask]

        # Аналитический Тета-фильтр Гильберта (отрицательные частоты = 0)
        f_theta = (torch.exp(-0.5 * ((freqs - 5.5) / 1.8)**2) * 2.0).view(1, 1, BUF_LEN)
        f_theta[:, :, freqs < 0] = 0.0

        # Аналитические Гамма-фильтры Гильберта (32 полосы, отрицательные частоты = 0)
        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(1, NUM_FREQS, 1, 1)
        freqs_4d = freqs.view(1, 1, 1, BUF_LEN)
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(1, NUM_FREQS, 1, 1)

        streams = resolve_byprop('type', 'EEG', timeout=0.3)
        inlets = [StreamInlet(s, max_buflen=1, max_chunklen=CHUNK_SIZE, recover=True) for s in streams[:NUM_DEVICES]]
        
        t_sim = 0.0
        smoothed_theta = 5.5
        dev_phase = torch.linspace(0, math.pi, NUM_DEVICES, device=DEVICE).view(NUM_DEVICES, 1, 1)
        ch_phase = torch.linspace(0, 2 * math.pi, NUM_CHANNELS, device=DEVICE).view(1, NUM_CHANNELS, 1)

        sh_vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        sh_vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        sh_tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        sh_gx = np.frombuffer(self.shm['gx'].get_obj(), dtype=np.float64).reshape(4, NUM_FREQS)
        sh_gy = np.frombuffer(self.shm['gy'].get_obj(), dtype=np.float64).reshape(4, NUM_FREQS)
        sh_iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_DEVICES, NUM_FREQS, NUM_PAIRS)

        while self.shm['is_running'].value:
            num_live = len(inlets)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            if is_real:
                pulled = False
                for i, inlet in enumerate(inlets):
                    chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=CHUNK_SIZE)
                    if chunk:
                        arr = torch.from_numpy(np.array(chunk, dtype=np.float32).T).to(DEVICE)
                        n = arr.shape[1]
                        eeg_ring[i] = torch.roll(eeg_ring[i], -n, dims=1)
                        eeg_ring[i, :, -n:] = arr
                        pulled = True
                if not pulled:
                    time.sleep(0.001)
                    continue
            else:
                t_sim += 0.015
                sim_th = 5.2 + 0.9 * math.sin(t_sim * 0.3) + 0.4 * math.sin(t_sim * 0.8)
                t_sub = torch.arange(CHUNK_SIZE, device=DEVICE, dtype=torch.float32) / FS
                sim_chunk = torch.sin(2 * math.pi * sim_th * (t_sub + t_sim) + ch_phase + dev_phase)
                eeg_ring = torch.roll(eeg_ring, -CHUNK_SIZE, dims=2)
                eeg_ring[:, :, -CHUNK_SIZE:] = sim_chunk

            with torch.inference_mode():
                centered = eeg_ring - torch.mean(eeg_ring, dim=2, keepdim=True)
                # Комплексное БПФ размера 256
                fft_raw = torch.fft.fft(centered, dim=-1)

                # 🔬 Непрерывный спектральный центр тяжести Теты (Sub-Bin Centroid)
                pwr_theta = torch.sum(torch.abs(fft_raw[0, :, theta_mask])**2, dim=0) # Размер [N_theta_bins]
                total_th_pwr = torch.sum(pwr_theta)
                
                if total_th_pwr > 1e-6:
                    spectral_centroid = torch.sum(pwr_theta * theta_freqs_slice) / total_th_pwr
                    raw_th_hz = float(torch.clamp(spectral_centroid, 4.2, 6.4).item())
                else:
                    raw_th_hz = 5.5

                # Плавный фильтр частоты Теты
                smoothed_theta = smoothed_theta * 0.92 + raw_th_hz * 0.08
                self.shm['theta_freq'].value = smoothed_theta

                # Комплексный аналитический фазор Теты
                Z_theta = torch.fft.ifft(fft_raw * f_theta, dim=-1) # complex64
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                phi_theta = torch.angle(torch.mean(P_theta, dim=1, keepdim=True)).unsqueeze(1)

                # Комплексные аналитические фазоры Гаммы
                fft_exp = fft_raw.unsqueeze(1)
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1) # complex64
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

                p_diff = phi_theta[:, :, :, -1:] - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                # Мнимая часть кросс-фазора (знак когерентности сохранен!)
                cg_gamma = P_gamma[:, :, I_GPU, -1] * torch.conj(P_gamma[:, :, J_GPU, -1])
                psi_field = cg_gamma * w.squeeze(-1)
                gamma_120 = torch.imag(psi_field) # [NUM_DEVICES, 32, 120] -> float32

                # Векторы потока
                vx = torch.sum(gamma_120[:, -1, :] * DX_GPU, dim=-1) * 3.5
                vy = torch.sum(gamma_120[:, -1, :] * DY_GPU, dim=-1) * 3.5
                tq = torch.sum(gamma_120[:, -1, :] * TQ_GPU, dim=-1) * 4.0

                traj_x = torch.sum(gamma_120 * DX_GPU.unsqueeze(1), dim=-1) * 0.5
                traj_y = torch.sum(gamma_120 * DY_GPU.unsqueeze(1), dim=-1) * 0.5

                # Обновление Shared Memory
                np.copyto(sh_vx, vx.cpu().numpy())
                np.copyto(sh_vy, vy.cpu().numpy())
                np.copyto(sh_tq, tq.cpu().numpy())
                np.copyto(sh_gx, traj_x.cpu().numpy())
                np.copyto(sh_gy, traj_y.cpu().numpy())
                np.copyto(sh_iplv, gamma_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self):
        self.roles = ["FCz (Macro/Music)", "Pz (Spatial)", "Oz (Sensory)", "Cz (Motor)"]
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.8),
            'theta_freq': mp.Value('d', 5.5),
            'vx': mp.Array('d', NUM_DEVICES),
            'vy': mp.Array('d', NUM_DEVICES),
            'tq': mp.Array('d', NUM_DEVICES),
            'gx': mp.Array('d', NUM_DEVICES * NUM_FREQS),
            'gy': mp.Array('d', NUM_DEVICES * NUM_FREQS),
            'iplv': mp.Array('d', NUM_DEVICES * NUM_FREQS * NUM_PAIRS)
        }
        self._vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        self._vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        self._tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        self._gx = np.frombuffer(self.shm['gx'].get_obj(), dtype=np.float64).reshape(4, NUM_FREQS)
        self._gy = np.frombuffer(self.shm['gy'].get_obj(), dtype=np.float64).reshape(4, NUM_FREQS)
        self._iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_DEVICES, NUM_FREQS, NUM_PAIRS)
        self.process = GPU_Daemon_Process(self.shm)

    def start(self): self.process.start()
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join()

    def get_frame(self) -> MultimodalFrame:
        nodes = []
        for i in range(4):
            traj_32 = np.stack([self._gx[i], self._gy[i]], axis=-1)
            nodes.append(NodeState(
                name=self.roles[i], device_id=i, vx=self._vx[i], vy=self._vy[i], tq=self._tq[i],
                thrust=math.hypot(self._vx[i], self._vy[i]), traj_32=traj_32, iplv_32=self._iplv[i]
            ))
        return MultimodalFrame(
            fcz_macro=nodes[0], pz_spatial=nodes[1], oz_sensory=nodes[2], cz_motor=nodes[3],
            theta_freq=self.shm['theta_freq'].value, theta_sync=self.shm['theta_sync'].value,
            is_real=self.shm['is_real'].value, num_live=self.shm['num_live'].value
        )
