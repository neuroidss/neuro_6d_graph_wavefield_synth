#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: PURE TENSOR CONTINUOUS PSYTRANCE SUITE (v74)
- Непрерывный темп (Continuous Theta Slaving).
- 100% Линейная тензорная интерполяция на CUDA (Без щелчков и цифрового шума).
- 100% Векторизованный расчет 32 точек Тора на GPU (Без циклов в Python).
- Чистый, плотный, аналоговый Full-On Psytrance.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import math
import numpy as np
import pygame
import sounddevice as sd
import queue
import threading
import torch

from neuro_heterarchy_core import (
    HeterarchicalBrainEngine, NUM_CHANNELS, NUM_FREQS, NUM_PAIRS,
    COORDS_X, COORDS_Y, I_IDX, J_IDX
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WIDTH, HEIGHT = 1280, 800
CENTER_X, CENTER_Y = 400, 400
R_MAJOR, R_MINOR = 240.0, 100.0

SRC_X, SRC_Y = COORDS_X[I_IDX] / 13.0, COORDS_Y[I_IDX] / 13.0
DST_X, DST_Y = COORDS_X[J_IDX] / 13.0, COORDS_Y[J_IDX] / 13.0
DX_PAIRS = (DST_X - SRC_X).astype(np.float32)
DY_PAIRS = (DST_Y - SRC_Y).astype(np.float32)
MID_X_PAIRS = ((SRC_X + DST_X) * 0.5).astype(np.float32)
MID_Y_PAIRS = ((SRC_Y + DST_Y) * 0.5).astype(np.float32)

RADII = np.hypot(COORDS_X, COORDS_Y)
IS_INNER = RADII < 8.0

idx_inner_inner, idx_outer_outer, idx_inner_outer = [], [], []
for p in range(NUM_PAIRS):
    ch_i, ch_j = I_IDX[p], J_IDX[p]
    if IS_INNER[ch_i] and IS_INNER[ch_j]: idx_inner_inner.append(p)
    elif not IS_INNER[ch_i] and not IS_INNER[ch_j]: idx_outer_outer.append(p)
    else: idx_inner_outer.append(p)

IDX_KICK = torch.tensor(idx_inner_inner, device=DEVICE, dtype=torch.long)
IDX_PAD  = torch.tensor(idx_outer_outer, device=DEVICE, dtype=torch.long)
IDX_ACID = torch.tensor(idx_inner_outer, device=DEVICE, dtype=torch.long)

# GPU-тензоры топологии для мгновенного векторного расчета координат
DX_GPU = torch.from_numpy(DX_PAIRS).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIRS).to(DEVICE)
MID_X_GPU = torch.from_numpy(MID_X_PAIRS).to(DEVICE)
MID_Y_GPU = torch.from_numpy(MID_Y_PAIRS).to(DEVICE)

SCALE_26MM = 2.0
RAW_INTERVALS = ((DX_PAIRS / SCALE_26MM) * 12.0 * 1.5 + (DY_PAIRS / SCALE_26MM) * 7.0).astype(np.float32)
CONSONANT_SCALE = np.array([-12, -11, -8, -7, -5, 0, 1, 4, 5, 7, 8, 12, 13, 16], dtype=np.float32)
QUANTIZED_INTERVALS = np.array([CONSONANT_SCALE[np.argmin(np.abs(CONSONANT_SCALE - val))] for val in RAW_INTERVALS], dtype=np.float32)

def torus_to_3d(theta, phi):
    return (R_MAJOR + R_MINOR * math.cos(phi)) * math.cos(theta), (R_MAJOR + R_MINOR * math.cos(phi)) * math.sin(theta), R_MINOR * math.sin(phi)

def project_3d(x, y, z, pitch=0.85, yaw=0.0):
    x1, y1 = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    y2, z2 = y1 * math.cos(pitch) - z * math.sin(pitch), y1 * math.sin(pitch) + z * math.cos(pitch)
    return int(CENTER_X + x1), int(CENTER_Y - y2), z2

# ==============================================================================
# ⚡ STUDIO CONTINUOUS TENSOR ENGINE (100% CUDA)
# ==============================================================================
class StudioContinuousEngine:
    def __init__(self, sr=SAMPLE_RATE, device=DEVICE):
        self.sr, self.block_size, self.device = sr, BLOCK_SIZE, device
        self.rfft_freqs = torch.fft.rfftfreq(self.block_size, d=1.0/self.sr, device=self.device)
        
        self.roots_midi = np.array([48, 55, 50, 57, 52, 47, 54, 49, 56, 51, 46, 53], dtype=np.float32)
        self.roots_hz_gpu = torch.from_numpy(440.0 * (2.0 ** ((self.roots_midi - 69.0) / 12.0))).to(self.device)

        self.intervals_gpu = torch.from_numpy(QUANTIZED_INTERVALS).to(self.device, dtype=torch.float32).unsqueeze(0)
        self.pan_l = torch.from_numpy(1.0 - MID_X_PAIRS * 0.5).to(self.device, dtype=torch.float32).unsqueeze(0)
        self.pan_r = torch.from_numpy(1.0 + MID_X_PAIRS * 0.5).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.osc_phases = torch.zeros((1, NUM_PAIRS), device=self.device, dtype=torch.float32) 
        self.kick_phases = torch.zeros((1, 6), device=self.device, dtype=torch.float32) 
        self.bass_phases = torch.zeros((1, 6), device=self.device, dtype=torch.float32) 
        
        self.theta_phase = 0.0
        self.prev_step = -1
        
        self.delay_buffer = torch.zeros((2, self.sr), device=self.device, dtype=torch.float32)
        self.delay_ptr = 0

        self.biological_theta_hz = 5.5 
        self.iplv_tensor = torch.zeros((NUM_FREQS, NUM_PAIRS), device=self.device, dtype=torch.float32)
        self.theta_pos, self.phi_pos = 0.0, 0.0
        self.target_rx, self.target_ry = 0.0, 0.0

        self.t_vec = torch.arange(self.block_size, device=self.device, dtype=torch.float32)
        self.param_lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=32)

    def update_state(self, theta_rad, phi_rad, iplv_32, theta_hz, rx, ry):
        with self.param_lock:
            self.theta_pos, self.phi_pos = theta_rad % (2.0 * math.pi), phi_rad % (2.0 * math.pi)
            self.iplv_tensor.copy_(torch.from_numpy(np.abs(iplv_32)))
            self.biological_theta_hz += (theta_hz - self.biological_theta_hz) * 0.1
            self.target_rx, self.target_ry = rx, ry

    def render_block(self):
        with self.param_lock:
            cur_th, cur_ph = self.theta_pos, self.phi_pos
            iplv = self.iplv_tensor 
            live_theta_hz = max(4.0, min(6.5, self.biological_theta_hz))
            rx_val, ry_val = abs(self.target_rx), self.target_ry

        with torch.inference_mode():
            # 1. Потоковое Тета-время
            theta_inc = 2.0 * math.pi * live_theta_hz / self.sr
            t_phase_accum = self.theta_phase + self.t_vec * theta_inc
            self.theta_phase = (t_phase_accum[-1].item() + theta_inc) % (2.0 * math.pi)
            
            theta_norm = (t_phase_accum / (2.0 * math.pi)) % 1.0

            # 2. Time-Warping через ry
            warp_factor = 2.0 ** -ry_val 
            warped_norm = theta_norm ** warp_factor 

            # 🔬 ЛИНЕЙНАЯ ИНТЕРПОЛЯЦИЯ 32 СРЕЗОВ (Убирает щелчки ступеньки)
            float_idx = warped_norm * 31.0
            idx_0 = float_idx.long().clamp(0, 30)
            idx_1 = idx_0 + 1
            alpha = (float_idx - idx_0.float()).unsqueeze(1) # [1024, 1]
            tensor_stream = iplv[idx_0, :] * (1.0 - alpha) + iplv[idx_1, :] * alpha # [1024, 120] C0-smooth

            mask_past    = torch.clamp(1.0 - (warped_norm / 0.33), 0.0, 1.0)
            mask_present = torch.clamp(1.0 - torch.abs(warped_norm - 0.5) / 0.33, 0.0, 1.0)
            mask_future  = torch.clamp((warped_norm - 0.66) / 0.34, 0.0, 1.0)

            # 16-дольная ритмика (8 шестнадцатых на слог)
            grid_8 = (theta_norm * 8.0)
            step_16th = grid_8.long() % 8
            sub_16th = grid_8 % 1.0
            
            curr_step = step_16th[0].item()
            if curr_step in (0, 4) and curr_step != self.prev_step:
                self.kick_phases.zero_()
            self.prev_step = curr_step
            
            is_kick = (step_16th % 4 == 0).float()
            is_bass = (step_16th % 4 != 0).float()
            is_open_hat = ((step_16th == 2) | (step_16th == 6)).float()
            is_closed_hat = ((step_16th == 1) | (step_16th == 3) | (step_16th == 5) | (step_16th == 7)).float()

            theta_diff = (torch.arange(12, device=self.device) * (2.0 * math.pi / 12.0) - cur_th + math.pi) % (2.0 * math.pi) - math.pi
            root_amps = torch.clamp(1.0 - torch.abs(theta_diff) / (math.pi / 3.0), 0.0, 1.0) ** 2
            base_f0 = self.roots_hz_gpu[torch.argmax(root_amps)]

            midi_notes = 48.0 + (base_f0 / 130.0) * 12.0 + self.intervals_gpu
            freqs_120 = 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0)) 
            incs_120 = 2.0 * math.pi * freqs_120 / self.sr 

            # 3. KICK & SUB-BASS (6 связей Ядра)
            core_tensor = tensor_stream[:, IDX_KICK]
            core_power = torch.log1p(3.0 * core_tensor) / math.log(4.0)

            kick_click_env = torch.exp(-sub_16th * 40.0) * is_kick
            kick_body_env  = torch.exp(-sub_16th * 6.0) * is_kick
            kick_freq = 48.0 + 2800.0 * kick_click_env
            k_incs = (2.0 * math.pi * kick_freq / self.sr).unsqueeze(1)

            k_phase_track = self.kick_phases + torch.cumsum(k_incs, dim=0)
            self.kick_phases = (k_phase_track[-1:, :] % (2.0 * math.pi))

            kick_waves = torch.sin(k_phase_track) * core_power
            kick_raw = torch.sum(kick_waves, dim=1) * kick_body_env * (0.9 / math.sqrt(6.0))
            kick_audio = torch.tanh(kick_raw * 1.8) * 0.85

            # Rolling Sub-Bass (45-65 Hz)
            bass_env = torch.exp(-sub_16th * 12.0) * is_bass
            b_incs = (2.0 * math.pi * (base_f0 / 2.0) / self.sr)
            b_phase_track = self.bass_phases + b_incs * self.t_vec.unsqueeze(1)
            self.bass_phases = (b_phase_track[-1:, :] % (2.0 * math.pi))

            bass_saws = 2.0 * ((b_phase_track / (2.0 * math.pi)) % 1.0) - 1.0
            bass_waves = bass_saws * core_power * bass_env.unsqueeze(1) * (0.8 / math.sqrt(6.0))
            bass_sat = torch.tanh(torch.sum(bass_waves, dim=1) * 1.6)
            
            fc_bass = 70.0 + 1200.0 * bass_env.mean().item()
            H_bass = 1.0 / torch.sqrt(1.0 + (self.rfft_freqs / fc_bass) ** 8)
            bass_audio = torch.fft.irfft(torch.fft.rfft(bass_sat) * H_bass, n=self.block_size) * 0.75

            # 4. 66-VOICE TORUS PAD (66 связей Кольца)
            pad_tensor = tensor_stream[:, IDX_PAD]
            pad_incs = incs_120[:, IDX_PAD]

            pad_phases_track = self.osc_phases[:, IDX_PAD] + pad_incs * self.t_vec.unsqueeze(1)
            self.osc_phases[:, IDX_PAD] = (pad_phases_track[-1:, :] % (2.0 * math.pi))

            saw1 = 2.0 * ((pad_phases_track / (2.0 * math.pi)) % 1.0) - 1.0
            saw2 = 2.0 * (((pad_phases_track + 0.1) / (2.0 * math.pi)) % 1.0) - 1.0
            pad_raw_waves = (saw1 + saw2) * 0.5 

            pad_amps = (torch.log1p(2.5 * pad_tensor) / math.log(3.5)) * mask_present.unsqueeze(1)
            pad_waves = pad_raw_waves * pad_amps * (0.4 / math.sqrt(66.0))

            pad_l_raw = torch.sum(pad_waves * self.pan_l[:, IDX_PAD], dim=1)
            pad_r_raw = torch.sum(pad_waves * self.pan_r[:, IDX_PAD], dim=1)

            fc_pad = 300.0 + 2500.0 * math.cos(cur_ph)**2
            H_pad = 1.0 / torch.sqrt(1.0 + (self.rfft_freqs / fc_pad) ** 4)
            ducking = 1.0 - 0.75 * kick_body_env
            pad_l = torch.fft.irfft(torch.fft.rfft(pad_l_raw) * H_pad, n=self.block_size) * ducking
            pad_r = torch.fft.irfft(torch.fft.rfft(pad_r_raw) * H_pad, n=self.block_size) * ducking

            # 5. 48-VOICE ACID SQUELCH (Плавный вейвфолдинг без артефактов)
            acid_tensor = tensor_stream[:, IDX_ACID]
            acid_incs = incs_120[:, IDX_ACID] * 2.0 

            acid_phases_track = self.osc_phases[:, IDX_ACID] + acid_incs * self.t_vec.unsqueeze(1)
            self.osc_phases[:, IDX_ACID] = (acid_phases_track[-1:, :] % (2.0 * math.pi))

            raw_acid_saw = 2.0 * ((acid_phases_track / (2.0 * math.pi)) % 1.0) - 1.0
            acid_folded = torch.sin(raw_acid_saw * (1.0 + acid_tensor * 4.0))
            
            acid_env = torch.exp(-sub_16th * 8.0).unsqueeze(1) * mask_future.unsqueeze(1)
            acid_amps = (torch.log1p(2.5 * acid_tensor) / math.log(3.5)) * acid_env
            acid_waves = acid_folded * acid_amps * (0.5 / math.sqrt(48.0))

            acid_l_raw = torch.sum(acid_waves * self.pan_l[:, IDX_ACID], dim=1)
            acid_r_raw = torch.sum(acid_waves * self.pan_r[:, IDX_ACID], dim=1)

            fc_acid_scalar = 150.0 + 5500.0 * rx_val
            Q = 1.5 + rx_val * 6.0
            H_acid_raw = 1.0 / torch.sqrt((1.0 - (self.rfft_freqs / fc_acid_scalar)**2)**2 + (self.rfft_freqs / (fc_acid_scalar * Q))**2 + 1e-5)
            H_acid = H_acid_raw / math.sqrt(Q)
            
            acid_l = torch.fft.irfft(torch.fft.rfft(acid_l_raw) * H_acid, n=self.block_size) * (0.6 * rx_val) * ducking
            acid_r = torch.fft.irfft(torch.fft.rfft(acid_r_raw) * H_acid, n=self.block_size) * (0.6 * rx_val) * ducking

            # 6. Хэты
            early_energy = torch.mean(tensor_stream[0:10, :]).item()
            hat_env = (torch.exp(-sub_16th * 14.0) * is_open_hat + torch.exp(-sub_16th * 45.0) * is_closed_hat) * min(1.0, early_energy * 2.0)
            hat_noise = torch.randn(self.block_size, device=self.device)
            H_hat = 1.0 - 1.0 / torch.sqrt(1.0 + (self.rfft_freqs / 8000.0) ** 8)
            hat_audio = torch.fft.irfft(torch.fft.rfft(hat_noise * hat_env) * H_hat, n=self.block_size) * 0.9

            # 7. Ping-Pong Delay
            delay_samples = int(self.sr * 0.375 / live_theta_hz)
            read_ptr = (self.delay_ptr - delay_samples) % self.sr
            idx_vec = (torch.arange(self.block_size, device=self.device) + read_ptr) % self.sr
            delayed_l = self.delay_buffer[0, idx_vec]
            delayed_r = self.delay_buffer[1, idx_vec]

            send_l = acid_l * 0.4 + pad_l * 0.3
            send_r = acid_r * 0.4 + pad_r * 0.3
            write_idx = (torch.arange(self.block_size, device=self.device) + self.delay_ptr) % self.sr
            
            self.delay_buffer[0, write_idx] = torch.tanh(send_l + delayed_r * 0.35)
            self.delay_buffer[1, write_idx] = torch.tanh(send_r + delayed_l * 0.35)
            self.delay_ptr = (self.delay_ptr + self.block_size) % self.sr

            mix_l = kick_audio + bass_audio + hat_audio + pad_l + acid_l + delayed_l * 0.5 * ducking
            mix_r = kick_audio + bass_audio + hat_audio + pad_r + acid_r + delayed_r * 0.5 * ducking

            out = torch.stack([mix_l, mix_r], dim=1)
            out = torch.tanh(out * 0.85) * 0.85
            return out.cpu().numpy()

def audio_thread_loop(synth, stop_event):
    def cb(outdata, frames, time_info, status):
        try: outdata[:] = synth.audio_queue.get_nowait()
        except queue.Empty: outdata.fill(0)
    for _ in range(8): synth.audio_queue.put(synth.render_block())
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=cb, blocksize=BLOCK_SIZE, dtype='float32'):
        while not stop_event.is_set():
            if synth.audio_queue.qsize() < 8: synth.audio_queue.put(synth.render_block())
            else: pygame.time.wait(2)

# ==============================================================================
# 🎮 MAIN VISUALIZER (100% GPU VECTORIZED COORDINATES)
# ==============================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Pure Tensor Continuous Psytrance Suite (v74)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14, bold=True)
    font_lg = pygame.font.SysFont("consolas", 18, bold=True)

    engine = HeterarchicalBrainEngine()
    engine.start()

    synth = StudioContinuousEngine()
    stop_event = threading.Event()
    t_audio = threading.Thread(target=audio_thread_loop, args=(synth, stop_event))
    t_audio.start()

    operating_mode = 1  
    avatar_theta, avatar_phi = 0.0, 0.0
    torus_yaw = 0.0

    KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
    QUALITIES = [("Phrygian", 0.0), ("Harmonic Min", 0.5*math.pi), ("Minor", math.pi), ("Diminished", 1.5*math.pi)]

    running = True
    try:
        while running:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: operating_mode = (operating_mode + 1) % 2

            frame = engine.get_frame()
            node = frame.fcz_macro
            ax = node.gamepad_axes
            active_iplv = node.iplv_32 # [32, 120]
            live_theta_hz = frame.theta_freq

            # 🔬 100% GPU РАСЧЕТ ВСЕХ 32 АБСОЛЮТНЫХ ТОЧЕК В ОДИН ТЕНЗОР (БЕЗ ЦИКЛОВ В PYTHON!)
            with torch.inference_mode():
                iplv_gpu = torch.from_numpy(active_iplv).to(DEVICE)
                slice_pwr = torch.abs(iplv_gpu) + 1e-6
                sum_pwr = torch.sum(slice_pwr, dim=1, keepdim=True)
                
                cx_all = torch.sum(slice_pwr * MID_X_GPU, dim=1, keepdim=True) / sum_pwr
                cy_all = torch.sum(slice_pwr * MID_Y_GPU, dim=1, keepdim=True) / sum_pwr
                vx_all = torch.sum(iplv_gpu * DX_GPU, dim=1, keepdim=True) / sum_pwr
                vy_all = torch.sum(iplv_gpu * DY_GPU, dim=1, keepdim=True) / sum_pwr
                
                th_all = (torch.atan2(cy_all + vy_all, cx_all + vx_all) % (2.0 * math.pi)).squeeze(1).cpu().numpy()
                rad_all = torch.hypot(cx_all + vx_all, cy_all + vy_all) * 2.0
                ph_all = ((rad_all * math.pi) % (2.0 * math.pi)).squeeze(1).cpu().numpy()

            head_th, head_ph = th_all[-1], ph_all[-1]
            glide_speed = 4.0 * (1.0 + max(0.0, ax.ry))
            avatar_theta += ((head_th - avatar_theta + math.pi) % (2.0 * math.pi) - math.pi) * glide_speed * dt
            avatar_phi   += ((head_ph - avatar_phi + math.pi) % (2.0 * math.pi) - math.pi) * glide_speed * dt
            avatar_theta = avatar_theta % (2.0 * math.pi)
            avatar_phi   = avatar_phi % (2.0 * math.pi)

            synth.update_state(avatar_theta, avatar_phi, active_iplv, live_theta_hz, ax.rx, ax.ry)

            screen.fill((10, 12, 18))

            # Сетка Тора
            for m in range(4):
                ph = m * (2.0 * math.pi / 4.0)
                pts = [project_3d(*torus_to_3d(t, ph), yaw=torus_yaw)[:2] for t in np.linspace(0, 2.0*math.pi, 60)]
                color = (60, 40, 70) if m != 0 else (120, 60, 160)
                pygame.draw.lines(screen, color, True, pts, 2 if m == 0 else 1)
                lx, ly, lz = project_3d(*torus_to_3d(math.pi/2.5, m * (2.0 * math.pi / 4.0)), yaw=torus_yaw)
                screen.blit(font.render(QUALITIES[m][0], True, (150, 150, 150)), (lx, ly))

            for i, key in enumerate(KEYS):
                th = i * (2.0 * math.pi / 12.0)
                pts = [project_3d(*torus_to_3d(th, p), yaw=torus_yaw)[:2] for p in np.linspace(0, 2.0*math.pi, 30)]
                pygame.draw.lines(screen, (35, 30, 45), True, pts, 1)
                tx, ty, tz = project_3d(*torus_to_3d(th, 0.0), yaw=torus_yaw)
                if tz > -10:
                    pygame.draw.circle(screen, (0, 200, 255), (tx, ty), 4)
                    screen.blit(font_lg.render(key, True, (0, 255, 200)), (tx + 8, ty - 10))

            # Кометный шлейф
            traj_3d_points = []
            for k in range(32):
                px, py, pz = project_3d(*torus_to_3d(th_all[k], ph_all[k]), yaw=torus_yaw)
                traj_3d_points.append((px, py, pz, th_all[k]))

            for k in range(31):
                p1 = traj_3d_points[k]
                p2 = traj_3d_points[k+1]
                
                d_th_check = abs(p1[3] - p2[3])
                if d_th_check < math.pi and p1[2] > -60 and p2[2] > -60:
                    if k < 11:
                        prog = k / 10.0
                        r, g, b = 0, int(120*(1-prog) + 255*prog), int(255*(1-prog) + 220*prog)
                    elif k < 22:
                        prog = (k - 11) / 10.0
                        r, g, b = int(255*prog), int(255*(1-prog) + 215*prog), int(180*(1-prog))
                    else:
                        prog = (k - 22) / 9.0
                        r, g, b = 255, int(140*(1-prog) + 20*prog), int(220*prog)
                    
                    col = (r, g, b)
                    thickness = max(1, int(1 + (k / 31.0) * 4))
                    pygame.draw.line(screen, col, p1[:2], p2[:2], thickness)
                    pygame.draw.circle(screen, col, p1[:2], max(1, int(1 + (k / 31.0) * 3)))

            # Аватар
            asx, asy, asz = project_3d(*torus_to_3d(avatar_theta, avatar_phi), yaw=torus_yaw)
            radius = max(3, int(15 + asz * 0.06)) 
            alpha_col = (255, 50, 150) if asz > 0 else (100, 20, 60)
            pygame.draw.circle(screen, alpha_col, (asx, asy), radius)
            pygame.draw.circle(screen, (255, 255, 255), (asx, asy), max(2, radius // 3))

            # Панели
            PANEL_X = 850
            sdr_y = 40
            screen.blit(font_lg.render("STREAMING 120-EDGE MATRIX", True, (0, 200, 255)), (PANEL_X, sdr_y))
            pygame.draw.rect(screen, (20, 25, 35), (PANEL_X, sdr_y + 30, 400, 150))
            pygame.draw.rect(screen, (0, 100, 150), (PANEL_X, sdr_y + 30, 400, 150), 1)
            
            mean_iplv = np.mean(np.abs(active_iplv), axis=0)
            max_v = np.max(mean_iplv) + 1e-6
            for i in range(120):
                val = mean_iplv[i] / max_v
                h = int(val * 140)
                if i in idx_inner_inner: col = (255, 80, 80)
                elif i in idx_outer_outer: col = (80, 255, 120)
                else: col = (80, 150, 255)
                pygame.draw.rect(screen, col, (PANEL_X + 10 + i*3, sdr_y + 30 + 145 - h, 2, h))

            ry_l = sdr_y + 210
            screen.blit(font_lg.render("CONTINUOUS SDR NEUROFEEDBACK", True, (255, 100, 200)), (PANEL_X, ry_l))
            pygame.draw.rect(screen, (35, 20, 25), (PANEL_X, ry_l + 30, 400, 180))
            pygame.draw.rect(screen, (150, 50, 100), (PANEL_X, ry_l + 30, 400, 180), 1)
            
            th_idx = int(round((avatar_theta / (2.0 * math.pi)) * 12.0)) % 12
            ph_idx = int(round((avatar_phi / (2.0 * math.pi)) * 4.0)) % 4
            current_bpm = live_theta_hz * 30.0 
            
            screen.blit(font_lg.render(f"KEY: {KEYS[th_idx]} {QUALITIES[ph_idx][0]}", True, (0, 255, 100)), (PANEL_X + 15, ry_l + 45))
            screen.blit(font_lg.render(f"THETA FREQ: {live_theta_hz:.2f} Hz -> {current_bpm:.0f} BPM", True, (255, 255, 255)), (PANEL_X + 15, ry_l + 70))
            
            screen.blit(font.render("• Cyan Tail (0..10)  -> Continuous Sub-Kick", True, (0, 255, 200)), (PANEL_X + 15, ry_l + 105))
            screen.blit(font.render("• Gold Body (11..21) -> 66-Voice Torus Pad", True, (255, 220, 50)), (PANEL_X + 15, ry_l + 130))
            screen.blit(font.render("• Pink Head (22..31) -> Smooth Wavefold Squelch", True, (255, 50, 150)), (PANEL_X + 15, ry_l + 155))

            ui_y = HEIGHT - 60
            pygame.draw.rect(screen, (15, 12, 22), (0, ui_y, WIDTH, 60))
            pygame.draw.line(screen, (0, 255, 200), (0, ui_y), (WIDTH, ui_y), 3)
            screen.blit(font_lg.render("NEUROCANVAS V74 | CONTINUOUS TENSOR PSYTRANCE | [SPACE] Toggle", True, (255, 255, 255)), (20, ui_y + 20))

            pygame.display.flip()

    finally:
        stop_event.set()
        t_audio.join()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
