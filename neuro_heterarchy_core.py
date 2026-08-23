#!/usr/bin/env python3
"""
🧠 NEURO-HETERARCHY CORE 2.0 (SINGLE SOURCE OF TRUTH)
Единый центральный CUDA-демон для всей экосистемы NeuroCanvas.
- Батчированные вычисления 4 устройств и 32 частот.
- Zero-Copy выгрузка в ОЗУ: 4-Axis Gamepad + Полный тензор iPLV (32x120).
"""

import os
import time
import math
import ctypes
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from pylsl import StreamInlet, resolve_byprop

FS = 250.0
BUF_SIZE = 256
NUM_CHANNELS = 16
NUM_DEVICES = 4
NUM_FREQS = 32
NUM_PAIRS = 120

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIR = COORDS_X[J_IDX] - COORDS_X[I_IDX]
DY_PAIR = COORDS_Y[J_IDX] - COORDS_Y[I_IDX]
TQ_MULT = (COORDS_X[I_IDX] * DY_PAIR - COORDS_Y[I_IDX] * DX_PAIR) / 100.0
SCALE_28_120 = 28.0 / 120.0

np.random.seed(42)
PROJ_MATRICES = np.stack([np.linalg.qr(np.random.randn(NUM_PAIRS, NUM_PAIRS))[0][:2, :].T for _ in range(NUM_DEVICES)], axis=0)

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
    traj_32: np.ndarray       # [32, 2] 2D координаты
    iplv_32: np.ndarray       # [32, 120] Полный тензор фазового поля для SVD-клиентов
    jpca_state: np.ndarray
    now_s0: np.ndarray
    future_sN: np.ndarray
    
    @property
    def gamepad_axes(self) -> UniversalGamepadAxes:
        if self.device_id == 3:
            b_len = math.hypot(self.vx, self.vy) + 1e-6
            lx = float(self.vx / b_len if b_len > 1.0 else self.vx)
            ly = float(self.vy / b_len if b_len > 1.0 else self.vy)
            rx = float(np.clip(self.tq * 0.8, -1.0, 1.0))
            ry = 0.0
            return UniversalGamepadAxes(lx, -ly, rx, ry)

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
        mid_x = self.traj_32[mid_idx, 0] - base_x
        mid_y = self.traj_32[mid_idx, 1] - base_y
        
        len_past = math.hypot(mid_x, mid_y)
        len_future = math.hypot(end_x - mid_x, end_y - mid_y)
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
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        print(f"[CORE ENGINE] Pure GPU Batched Daemon Active on {DEVICE}...")

        freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        
        notch = torch.ones_like(freqs)
        notch[(torch.abs(freqs) >= 48.0) & (torch.abs(freqs) <= 52.0)] = 0.0
        notch[(torch.abs(freqs) >= 98.0) & (torch.abs(freqs) <= 102.0)] = 0.0
        notch = notch.view(1, 1, BUF_SIZE)

        f_beta = (torch.exp(-0.5 * ((freqs - 22.0) / 8.0)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_beta[:, :, freqs < 0] = 0.0

        f_theta = (torch.exp(-0.5 * ((freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_theta[:, :, freqs < 0] = 0.0

        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(1, NUM_FREQS, 1, 1)
        freqs_4d = freqs.view(1, 1, 1, BUF_SIZE)
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(1, NUM_FREQS, 1, 1)

        I_GPU = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
        J_GPU = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)
        DX_GPU = torch.from_numpy(DX_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        DY_GPU = torch.from_numpy(DY_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        TQ_GPU = torch.from_numpy(TQ_MULT).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        PROJ_BATCH_GPU = torch.from_numpy(PROJ_MATRICES).to(DEVICE, dtype=torch.float32)

        M_skew = torch.tensor([[0,-2.4,0,0], [2.4,0,0,0], [0,0,0,-3.6], [0,0,3.6,0]], device=DEVICE, dtype=torch.float32)
        jpca_states = torch.tensor([[0.8,0,0.5,0] for _ in range(4)], device=DEVICE, dtype=torch.float32)

        streams = resolve_byprop('type', 'EEG', timeout=0.4)
        inlets = [StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True) for s in streams[:NUM_DEVICES]]
        
        raw_buffers = np.zeros((NUM_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)
        
        t_sim = 0.0
        ch_phase = torch.linspace(0, 2 * math.pi, 16, device=DEVICE).view(1, 16, 1)
        t_vec = torch.linspace(0, 1, 256, device=DEVICE).view(1, 1, 256)
        last_t = time.perf_counter()

        sh_vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        sh_vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        sh_tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        sh_jpca = np.frombuffer(self.shm['jpca'].get_obj(), dtype=np.float64).reshape(4, 4)
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
                    chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                    if chunk:
                        arr = np.array(chunk, dtype=np.float32).T
                        n = arr.shape[1]
                        if n >= BUF_SIZE: raw_buffers[i] = arr[:, -BUF_SIZE:]
                        else:
                            raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                            raw_buffers[i][:, -n:] = arr
                        pulled = True
                if not pulled:
                    time.sleep(0.001)
                    continue
                raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            else:
                t_sim += 0.015
                raw_buf_gpu = torch.sin(2*math.pi*6.0*t_vec + ch_phase + t_sim) + 0.5*torch.sin(2*math.pi*24.0*t_vec + ch_phase*2)

            now = time.perf_counter()
            dt = max(0.001, now - last_t)
            last_t = now

            with torch.inference_mode():
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                # 1. Бета-поток
                Z_beta = torch.fft.ifft(fft_clean * f_beta, dim=-1)
                P_beta = Z_beta / (torch.abs(Z_beta) + 1e-12)
                cg_beta = P_beta[:, I_GPU, :] * torch.conj(P_beta[:, J_GPU, :])
                iplv_beta = torch.mean(torch.imag(cg_beta), dim=-1)

                vx = torch.sum(iplv_beta * DX_GPU, dim=-1) * (SCALE_28_120 * 15.0)
                vy = torch.sum(iplv_beta * DY_GPU, dim=-1) * (SCALE_28_120 * 15.0)
                tq = torch.sum(iplv_beta * TQ_GPU, dim=-1) * (SCALE_28_120 * 18.0)

                # 2. Тета-Гамма VTE Поле
                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                mean_th_phasor = torch.mean(P_theta, dim=1, keepdim=True)
                phi_theta = torch.angle(mean_th_phasor).unsqueeze(1)
                
                # Синхронизация Тета
                sync_r = float(torch.mean(torch.abs(mean_th_phasor)).item())
                self.shm['theta_sync'].value = sync_r

                fft_exp = fft_clean.unsqueeze(1)
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

                p_diff = phi_theta - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_field = torch.sum(cg_gamma * w, dim=-1) # [4, 32, 120]
                
                # Anchor-referenced iPLV (vs slot 0)
                past_anchor = psi_field[:, 0:1, :]
                gamma_120 = torch.imag(psi_field * torch.conj(past_anchor)) # [4, 32, 120]

                # 2D Проекция для радаров
                traj_2d = torch.bmm(gamma_120, PROJ_BATCH_GPU) * 8.0

                # 3. jPCA
                omega_mod = (1.0 + tq * 2.0).view(4, 1)
                dx_jpca = torch.matmul(jpca_states, M_skew.T) * omega_mod * dt
                jpca_states += dx_jpca

                # Выгрузка в Shared Memory
                traj_cpu = traj_2d.cpu().numpy()
                np.copyto(sh_vx, vx.cpu().numpy())
                np.copyto(sh_vy, vy.cpu().numpy())
                np.copyto(sh_tq, tq.cpu().numpy())
                np.copyto(sh_jpca, jpca_states.cpu().numpy())
                np.copyto(sh_gx, traj_cpu[:, :, 0])
                np.copyto(sh_gy, traj_cpu[:, :, 1])
                np.copyto(sh_iplv, gamma_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self):
        self.roles = ["FCz (Macro/Music)", "Pz (Spatial/Drone)", "Oz (Sensory/Vision)", "Cz (Motor/SMA)"]
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.0),
            'theta_freq': mp.Value('d', 6.0),
            'vx': mp.Array('d', NUM_DEVICES),
            'vy': mp.Array('d', NUM_DEVICES),
            'tq': mp.Array('d', NUM_DEVICES),
            'jpca': mp.Array('d', NUM_DEVICES * 4),
            'gx': mp.Array('d', NUM_DEVICES * NUM_FREQS),
            'gy': mp.Array('d', NUM_DEVICES * NUM_FREQS),
            'iplv': mp.Array('d', NUM_DEVICES * NUM_FREQS * NUM_PAIRS)
        }
        self._vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        self._vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        self._tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        self._jpca = np.frombuffer(self.shm['jpca'].get_obj(), dtype=np.float64).reshape(4, 4)
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
            vx, vy = self._vx[i], self._vy[i]
            nodes.append(NodeState(
                name=self.roles[i], device_id=i,
                vx=vx, vy=vy, tq=self._tq[i], thrust=math.hypot(vx, vy),
                traj_32=traj_32,
                iplv_32=self._iplv[i].copy(), # [32, 120] Полный тензор фазового поля
                jpca_state=self._jpca[i],
                now_s0=traj_32[0], future_sN=traj_32[-1]
            ))
        return MultimodalFrame(
            fcz_macro=nodes[0], pz_spatial=nodes[1], oz_sensory=nodes[2], cz_motor=nodes[3],
            theta_freq=self.shm['theta_freq'].value, theta_sync=self.shm['theta_sync'].value,
            is_real=self.shm['is_real'].value, num_live=self.shm['num_live'].value
        )
