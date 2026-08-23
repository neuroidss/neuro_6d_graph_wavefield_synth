#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: MODAL PERCUSSION PSY-TORUS NAVIGATOR (GPU ACCELERATED)
CLOSED-LOOP 120-EDGE COHERENCE GRAPH & 3D TONAL TORUS SYNTHESIZER

THEORETICAL MODEL:
1. 3D Tonal Torus: Janata et al. (Science 2002, 10.1126/science.1076262)
2. Directed iPLV Graph: Bruña, Maestú, Pereda (J. Neural Eng. 2018, 10.1088/1741-2552/aacfe4)
3. Working Memory 2.0 (Theta-Gamma Code): Lisman & Jensen (Neuron 2013); Miller et al. (Neuron 2018)
4. Auditory Predictive Coding: Vuust et al. (Nat. Rev. Neurosci. 2022); Koelsch (Nat. Rev. Neurosci. 2014)

ACOUSTIC ARCHITECTURE:
- 120 links partitioned into 3 physical modal percussion registers (Shakers / FM Bongos / Snares).
- Dynamic Multi-Layer Psy-Kick (Beater Click + 48Hz Sub-Body) with sidechain ducking.
- Sagitta (rx) -> Tabla/Zap dynamic pitch bend (Acid Squelch / Q-resonance).
- Temporal Bias (ry) -> K-B-B-B Rolling Bass momentum & swing density.
- Seamless Style Morph [W/S]: Dark Psybient <=========> Full-On Psytrance.
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
    HeterarchicalBrainEngine, NUM_FREQS, NUM_PAIRS, 
    COORDS_X, COORDS_Y, I_IDX, J_IDX
)

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

# ==============================================================================
# 26-MM SENSOR TOPOLOGY (16 ELECTRODES, 120 DIRECTED EDGES)
# ==============================================================================
SRC_X, SRC_Y = COORDS_X[I_IDX], COORDS_Y[I_IDX]
DST_X, DST_Y = COORDS_X[J_IDX], COORDS_Y[J_IDX]

DX_PAIRS = DST_X - SRC_X
DY_PAIRS = DST_Y - SRC_Y
SCALE_26MM = 26.0 

# Physical euclidean length of each of the 120 pairs across the scalp (in mm)
PAIR_LENGTHS = np.hypot(DX_PAIRS, DY_PAIRS)

# Partition 120 links into 3 physical acoustic registers:
# 1. Short links (< 8 mm)       -> Glitch clicks, shakers (5-10 kHz)
MASK_SHORT_CPU = (PAIR_LENGTHS < 8.0).astype(np.float32)
# 2. Medium links (8 - 15 mm)   -> Tonal FM Bongos / Zaps / Acid Plucks
MASK_MED_CPU   = ((PAIR_LENGTHS >= 8.0) & (PAIR_LENGTHS < 15.0)).astype(np.float32)
# 3. Long links (>= 15 mm)      -> Snare / Clap / Noise transients
MASK_LONG_CPU  = (PAIR_LENGTHS >= 15.0).astype(np.float32)

RAW_INTERVALS = ((DX_PAIRS / SCALE_26MM) * 12.0 * 1.5 + (DY_PAIRS / SCALE_26MM) * 7.0).astype(np.float32)

# Phrygian / Euphoric Full-On harmonic pitch scale
CONSONANT_SCALE = np.array([-12, -5, 0, 3, 7, 10, 12, 14, 15, 19, 22, 24, 27, 31], dtype=np.float32)

QUANTIZED_INTERVALS = np.zeros(NUM_PAIRS, dtype=np.float32)
for p in range(NUM_PAIRS):
    idx = np.argmin(np.abs(CONSONANT_SCALE - RAW_INTERVALS[p]))
    QUANTIZED_INTERVALS[p] = CONSONANT_SCALE[idx]

MICRO_DETUNE_CPU = (RAW_INTERVALS - QUANTIZED_INTERVALS) * 0.12
MID_X_CPU = ((SRC_X + DST_X) * 0.5 / 13.0).astype(np.float32)

TORUS_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
CHORD_ROOTS = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=np.float32)
NUM_KEYS = 12

def midi_to_hz(midi_val):
    return 440.0 * (2.0 ** ((midi_val - 69.0) / 12.0))

# ==============================================================================
# GPU SYNTHESIZER: MODAL PERCUSSION & PSYBIENT ENGINE
# ==============================================================================
class ModalPercussionPsySynth:
    def __init__(self):
        self.sr = SAMPLE_RATE
        self.block_size = BLOCK_SIZE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AUDIO ENGINE] Modal Percussion GPU DSP initialized on: {self.device}")

        # 12 keys x 120 edges frequency grid [12, 120]
        freq_grid = np.zeros((NUM_KEYS, NUM_PAIRS), dtype=np.float32)
        for k in range(NUM_KEYS):
            root_k = 48.0 + CHORD_ROOTS[k]
            midi_notes = root_k + QUANTIZED_INTERVALS + MICRO_DETUNE_CPU
            freq_grid[k, :] = midi_to_hz(midi_notes)

        # Fundamental sub-bass for 12 keys (36 = C2)
        bass_freq = midi_to_hz(36.0 + CHORD_ROOTS)

        self.freq_grid_gpu = torch.from_numpy(freq_grid).to(self.device)
        self.bass_freq_gpu = torch.from_numpy(bass_freq).to(self.device)

        self.phase_incs_gpu = (2.0 * math.pi * self.freq_grid_gpu / self.sr).unsqueeze(-1)
        self.bass_incs_gpu  = (2.0 * math.pi * self.bass_freq_gpu / self.sr).unsqueeze(-1)

        self.dx_gpu = torch.from_numpy(DX_PAIRS).to(self.device, dtype=torch.float32)
        self.dy_gpu = torch.from_numpy(DY_PAIRS).to(self.device, dtype=torch.float32)
        self.mid_x_gpu = torch.from_numpy(MID_X_CPU).to(self.device, dtype=torch.float32).view(1, NUM_PAIRS, 1)

        # Percussion register masks [1, 120, 1]
        self.mask_short = torch.from_numpy(MASK_SHORT_CPU).to(self.device).view(1, NUM_PAIRS, 1)
        self.mask_med   = torch.from_numpy(MASK_MED_CPU).to(self.device).view(1, NUM_PAIRS, 1)
        self.mask_long  = torch.from_numpy(MASK_LONG_CPU).to(self.device).view(1, NUM_PAIRS, 1)

        self.theta_nodes_gpu = torch.linspace(0, 2.0 * math.pi, NUM_KEYS, device=self.device, dtype=torch.float32)
        self.step_rad = 2.0 * math.pi / NUM_KEYS

        # Per-sample vector grids
        self.t_vec_gpu = torch.arange(self.block_size, device=self.device, dtype=torch.float32).view(1, 1, self.block_size)
        self.bass_t_vec_gpu = torch.arange(self.block_size, device=self.device, dtype=torch.float32).view(1, self.block_size)
        self.t_ramp_gpu = torch.linspace(0.0, 1.0, self.block_size, device=self.device, dtype=torch.float32).view(1, 1, self.block_size)

        # 12 persistent phase banks
        self.bank_phases_gpu = torch.zeros((NUM_KEYS, NUM_PAIRS), device=self.device, dtype=torch.float32)
        self.bass_phases_gpu = torch.zeros(NUM_KEYS, device=self.device, dtype=torch.float32)
        self.groove_phase = 0.0
        self.kick_phase = 0.0

        self.prev_key_gains_gpu = torch.zeros(NUM_KEYS, device=self.device, dtype=torch.float32)
        self.prev_edge_amps_gpu = torch.zeros((1, NUM_PAIRS, 1), device=self.device, dtype=torch.float32)

        # Input tensor buffers
        self.target_iplv_gpu = torch.zeros((NUM_FREQS, NUM_PAIRS), device=self.device, dtype=torch.float32)
        self.target_traj_gpu = torch.zeros((NUM_FREQS, 2), device=self.device, dtype=torch.float32)
        
        self.target_theta = 0.0
        self.target_phi = 0.0
        self.curr_theta = 0.0
        self.curr_phi = 0.0
        self.live_theta_sync = 0.5
        self.style_morph = 0.0
        
        self.param_lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=32)

    def update_state(self, theta, phi, traj_32, iplv_32, theta_s, morph):
        with self.param_lock:
            self.target_theta = theta
            self.target_phi = phi
            self.target_traj_gpu.copy_(torch.from_numpy(traj_32))
            self.target_iplv_gpu.copy_(torch.from_numpy(np.abs(iplv_32)))
            self.live_theta_sync = float(np.clip(theta_s, 0.05, 1.0))
            self.style_morph = float(np.clip(morph, 0.0, 1.0))

    def render_block(self):
        alpha = 0.05
        with self.param_lock:
            d_th = (self.target_theta - self.curr_theta + math.pi) % (2 * math.pi) - math.pi
            self.curr_theta += d_th * alpha
            d_ph = (self.target_phi - self.curr_phi + math.pi) % (2 * math.pi) - math.pi
            self.curr_phi += d_ph * alpha
            
            theta_val = self.curr_theta
            traj = self.target_traj_gpu
            iplv = self.target_iplv_gpu
            th_sync = self.live_theta_sync
            morph = self.style_morph

        with torch.inference_mode():
            # ==================================================================
            # 1. TEMPORAL DYNAMICS STRUCTURE (S0 -> S8 -> S16 -> S24 -> S31)
            # ==================================================================
            s0 = traj[0]
            s8 = traj[8] - s0
            s16 = traj[16] - s0
            s24 = traj[24] - s0
            s31 = traj[31] - s0

            d_len = torch.hypot(s31[0], s31[1]) + 1e-6
            dir_x, dir_y = s31[0] / d_len, s31[1] / d_len

            # Dynamic step velocities along trajectory segments
            v0 = torch.hypot(s8[0], s8[1]) + 1e-5
            v1 = torch.hypot(s16[0] - s8[0], s16[1] - s8[1]) + 1e-5
            v2 = torch.hypot(s24[0] - s16[0], s24[1] - s16[1]) + 1e-5
            v3 = torch.hypot(s31[0] - s24[0], s31[1] - s24[1]) + 1e-5

            total_v = v0 + v1 + v2 + v3
            w_kick = float(torch.clamp(v0 / total_v * 1.8, 0.5, 1.4))
            w_b1   = float(torch.clamp(v1 / total_v * 1.8, 0.5, 1.4))
            w_b2   = float(torch.clamp(v2 / total_v * 1.8, 0.5, 1.4))
            w_b3   = float(torch.clamp(v3 / total_v * 1.8, 0.5, 1.4))

            # Dynamic musical tempo: scaled to 135 - 146 BPM
            live_bpm = 136.0 + float(torch.clamp(total_v * 2.5, 0.0, 10.0))
            beat_hz = live_bpm / 60.0 # Quarter-note beat frequency
            beat_inc = (2.0 * math.pi * beat_hz / self.sr)

            # Quarter-note phase vector [1024]
            beat_phase = (self.groove_phase + beat_inc * self.bass_t_vec_gpu.squeeze(0)) % (2.0 * math.pi)
            self.groove_phase = (self.groove_phase + beat_inc * self.block_size) % (2.0 * math.pi)
            beat_norm = beat_phase / (2.0 * math.pi) # 0.0 .. 1.0

            # 4 sixteenth-note sub-divisions per quarter beat
            sub_16th = (beat_norm * 4.0) % 1.0
            step_idx = (beat_norm * 4.0).long() % 4 # 0=Kick, 1=B1, 2=B2, 3=B3

            # ==================================================================
            # 2. MULTI-LAYER PSY-KICK (PUNCH CLICK + 48Hz SUB)
            # ==================================================================
            attack_ramp = torch.clamp(sub_16th * 12.0, 0.0, 1.0)
            
            is_kick = (step_idx == 0).float()
            kick_sub_env = attack_ramp * torch.exp(-sub_16th * 6.0) * is_kick
            kick_click = torch.clamp(sub_16th * 40.0, 0.0, 1.0) * torch.exp(-sub_16th * 28.0) * is_kick
            
            # Exponential pitch sweep: 180Hz -> 46Hz
            kick_freq = 46.0 + 135.0 * torch.exp(-sub_16th * 18.0)
            k_incs = 2.0 * math.pi * kick_freq / self.sr
            k_accum = torch.cumsum(k_incs, dim=0) + self.kick_phase
            self.kick_phase = (k_accum[-1].item()) % (2.0 * math.pi)
            
            fullon_kick = (torch.sin(k_accum) * kick_sub_env + kick_click * 0.30 * (torch.rand(self.block_size, device=self.device)*2-1)) * (0.85 * w_kick)
            sidechain_duck = 1.0 - 0.60 * kick_sub_env * morph

            # ==================================================================
            # 3. LEGATO ROLLING BASS (K-B-B-B)
            # ==================================================================
            decay_bass = torch.exp(-sub_16th * 4.6)
            is_b1 = (step_idx == 1).float() * w_b1
            is_b2 = (step_idx == 2).float() * w_b2
            is_b3 = (step_idx == 3).float() * w_b3
            
            bass_step_gain = is_b1 + is_b2 + is_b3
            rolling_bass_env = attack_ramp * decay_bass * bass_step_gain

            # ==================================================================
            # 4. 120 EDGES: MODAL PERCUSSION MATRIX (3 ACOUSTIC LAYERS)
            # ==================================================================
            low_g = torch.sum(iplv[:16, :], dim=0)
            high_g = torch.sum(iplv[16:, :], dim=0)
            total_pwr = (low_g + high_g) * 0.5
            norm_pwr = total_pwr / (torch.max(total_pwr) + 1e-5)
            companded_pwr = torch.log1p(4.0 * norm_pwr) / math.log(5.0)

            # Sagitta -> Modal Percussion Pitch Bend & Squelch (Tabla Zap)
            cross_t = torch.abs(dir_x * (self.dy_gpu / SCALE_26MM) - dir_y * (self.dx_gpu / SCALE_26MM))
            zap_pitch_bend = 1.0 + cross_t * (1.2 + 2.0 * morph)

            target_edge_amps = (0.20 + 0.80 * companded_pwr * zap_pitch_bend).view(1, NUM_PAIRS, 1)
            target_edge_amps = target_edge_amps * (0.24 / math.sqrt(NUM_PAIRS))

            edge_ramp = self.prev_edge_amps_gpu + (target_edge_amps - self.prev_edge_amps_gpu) * self.t_ramp_gpu
            self.prev_edge_amps_gpu.copy_(target_edge_amps)

            # Rhythmic envelope gating across 3 physical link classes:
            shaker_env = attack_ramp.view(1, 1, self.block_size) * torch.exp(-sub_16th * 9.0).view(1, 1, self.block_size)
            
            edge_step_mask = ((torch.arange(NUM_PAIRS, device=self.device).view(1, NUM_PAIRS, 1) % 4) == step_idx.view(1, 1, self.block_size)).float()
            bongo_env = attack_ramp.view(1, 1, self.block_size) * decay_bass.view(1, 1, self.block_size) * edge_step_mask
            
            is_snare_beat = (step_idx == 2).float().view(1, 1, self.block_size)
            snare_env = attack_ramp.view(1, 1, self.block_size) * torch.exp(-sub_16th * 6.0).view(1, 1, self.block_size) * is_snare_beat

            modal_perc_env = (self.mask_short * shaker_env * 0.4 + 
                              self.mask_med   * bongo_env  * 0.8 + 
                              self.mask_long  * snare_env  * 0.7)

            # ==================================================================
            # 5. POLYPHONIC TORUS RENDERING
            # ==================================================================
            d_th_all = torch.abs((theta_val - self.theta_nodes_gpu + math.pi) % (2.0 * math.pi) - math.pi)
            raw_gains = torch.clamp(1.0 - (d_th_all / self.step_rad), min=0.0, max=1.0)
            target_key_gains = torch.cos((1.0 - raw_gains) * (math.pi * 0.5))
            target_key_gains = target_key_gains / (torch.norm(target_key_gains) + 1e-6)

            left_pad = torch.zeros(self.block_size, device=self.device, dtype=torch.float32)
            right_pad = torch.zeros(self.block_size, device=self.device, dtype=torch.float32)
            left_perc = torch.zeros(self.block_size, device=self.device, dtype=torch.float32)
            right_perc = torch.zeros(self.block_size, device=self.device, dtype=torch.float32)
            bass_total = torch.zeros(self.block_size, device=self.device, dtype=torch.float32)

            active_mask = (self.prev_key_gains_gpu > 1e-4) | (target_key_gains > 1e-4)
            active_indices = torch.nonzero(active_mask).squeeze(-1)

            pan_l = (1.0 - self.mid_x_gpu * 0.4)
            pan_r = (1.0 + self.mid_x_gpu * 0.4)

            # FM Modulator scaling [1, 120, 1]
            fm_amt = (0.15 + 0.50 * morph + cross_t * 0.6).view(1, NUM_PAIRS, 1)

            for k in active_indices:
                g_start = self.prev_key_gains_gpu[k]
                g_end = target_key_gains[k]
                g_ramp_k = (g_start + (g_end - g_start) * self.t_ramp_gpu.squeeze(0)).unsqueeze(0)

                # 120 oscillators of the k-th harmonic key
                p_mat_k = self.bank_phases_gpu[k:k+1].unsqueeze(-1) + self.phase_incs_gpu[k:k+1] * self.t_vec_gpu
                
                # TORUS HARMONIC PAD (Continuous Foundation)
                pad_carrier = torch.sin(p_mat_k) + 0.12 * torch.sin(2.0 * p_mat_k)
                pad_wave = pad_carrier * (edge_ramp * g_ramp_k) * sidechain_duck.view(1, 1, self.block_size)
                left_pad += torch.sum(pad_wave * pan_l, dim=1).squeeze(0)
                right_pad += torch.sum(pad_wave * pan_r, dim=1).squeeze(0)

                # MODAL PERCUSSION (FM Zaps + Shakers + Bongos)
                fm_mod = torch.sin(2.0 * p_mat_k) * fm_amt
                perc_carrier = torch.sin(p_mat_k + fm_mod) + 0.20 * torch.sin(3.0 * p_mat_k)
                perc_wave = perc_carrier * (edge_ramp * modal_perc_env * g_ramp_k) * sidechain_duck.view(1, 1, self.block_size)
                
                left_perc += torch.sum(perc_wave * pan_l, dim=1).squeeze(0)
                right_perc += torch.sum(perc_wave * pan_r, dim=1).squeeze(0)

                # ROLLING BASS (K-B-B-B)
                b_mat_k = self.bass_phases_gpu[k] + self.bass_incs_gpu[k, 0] * self.bass_t_vec_gpu.squeeze(0)
                saw_bass = torch.sin(b_mat_k) + (0.20 + 0.40 * morph) * torch.sin(2.0 * b_mat_k) + (0.08 + 0.25 * morph) * torch.sin(3.0 * b_mat_k)
                
                b_env = (1.0 - morph) * 1.0 + morph * rolling_bass_env
                bass_k = saw_bass * (b_env * g_ramp_k.squeeze() * (0.24 + 0.16 * morph)) * sidechain_duck
                bass_total += bass_k

            self.prev_key_gains_gpu.copy_(target_key_gains)

            self.bank_phases_gpu = (self.bank_phases_gpu + self.phase_incs_gpu.squeeze(-1) * self.block_size) % (2.0 * math.pi)
            self.bass_phases_gpu = (self.bass_phases_gpu + self.bass_incs_gpu.squeeze(-1) * self.block_size) % (2.0 * math.pi)

            # ==================================================================
            # 6. MASTER MIX
            # ==================================================================
            synth_l = left_pad * 0.45 + left_perc * (0.55 * morph)
            synth_r = right_pad * 0.45 + right_perc * (0.55 * morph)
            
            kick_out = fullon_kick * morph
            
            out_l = synth_l * 0.65 + bass_total * 0.35 + kick_out
            out_r = synth_r * 0.65 + bass_total * 0.35 + kick_out

            out_l = torch.tanh(out_l * 0.90) * 0.72
            out_r = torch.tanh(out_r * 0.90) * 0.72

            stereo_gpu = torch.stack([out_l, out_r], dim=1)
            return stereo_gpu.cpu().numpy()

# ==============================================================================
# AUDIO STREAM THREAD
# ==============================================================================
def audio_thread_loop(synth, stop_event):
    def cb(outdata, frames, time_info, status):
        try:
            chunk = synth.audio_queue.get_nowait()
            outdata[:] = chunk
        except queue.Empty:
            outdata.fill(0)

    # Pre-fill audio queue
    for _ in range(12):
        synth.audio_queue.put(synth.render_block())

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, callback=cb, blocksize=BLOCK_SIZE):
        while not stop_event.is_set():
            if synth.audio_queue.qsize() < 12:
                synth.audio_queue.put(synth.render_block())
            else:
                pygame.time.wait(2)

# ==============================================================================
# MAIN TORUS NAVIGATION LOOP
# ==============================================================================
WIDTH, HEIGHT = 980, 780
CENTER_X, CENTER_Y = 320, 360

R_MAJOR = 160.0
R_MINOR = 65.0

def torus_to_3d(theta, phi):
    x = (R_MAJOR + R_MINOR * math.cos(phi)) * math.cos(theta)
    y = (R_MAJOR + R_MINOR * math.cos(phi)) * math.sin(theta)
    z = R_MINOR * math.sin(phi)
    return x, y, z

def project_3d(x, y, z, pitch=0.85, yaw=0.0):
    x1, y1 = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    x2, y2, z2 = x1, y1 * math.cos(pitch) - z * math.sin(pitch), y1 * math.sin(pitch) + z * math.cos(pitch)
    return int(CENTER_X + x2), int(CENTER_Y - y2), z2

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroCanvas: Modal Percussion Psy-Torus Engine")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 12, bold=True)
    font_lg = pygame.font.SysFont("consolas", 16, bold=True)

    engine = HeterarchicalBrainEngine()
    engine.start()

    synth = ModalPercussionPsySynth()
    stop_event = threading.Event()
    t_audio = threading.Thread(target=audio_thread_loop, args=(synth, stop_event))
    t_audio.start()

    theta_avatar = 0.0
    phi_avatar = 0.0
    torus_trail = []
    current_morph = 1.0

    running = True
    try:
        while running:
            dt = min(0.05, clock.tick(60) / 1000.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Smooth style morphing via [W / S] or [UP / DOWN]
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                current_morph = min(1.0, current_morph + 0.7 * dt)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                current_morph = max(0.0, current_morph - 0.7 * dt)

            frame = engine.get_frame()
            node = frame.fcz_macro
            ax = node.gamepad_axes

            force_x = ax.lx
            force_y = -ax.ly
            accel_mult = 1.0 + ax.ry * 0.5
            base_speed = 1.5 * accel_mult

            theta_avatar = (theta_avatar + force_x * base_speed * dt) % (2.0 * math.pi)
            phi_avatar   = (phi_avatar   + force_y * base_speed * dt) % (2.0 * math.pi)

            # Stream real-time tensors to GPU synth
            synth.update_state(
                theta=theta_avatar,
                phi=phi_avatar,
                traj_32=node.traj_32,
                iplv_32=node.iplv_32,
                theta_s=frame.theta_sync,
                morph=current_morph
            )

            torus_trail.append((theta_avatar, phi_avatar))
            if len(torus_trail) > 35:
                torus_trail.pop(0)

            # ------------------------------------------------------------------
            # RENDERING
            # ------------------------------------------------------------------
            screen.fill((6, 8, 12))

            # 1. 3D Tonal Torus Grid
            for k in range(12):
                th = k * (2.0 * math.pi / 12.0)
                pts = [project_3d(*torus_to_3d(th, p))[:2] for p in np.linspace(0, 2.0*math.pi, 20)]
                pygame.draw.lines(screen, (20, 28, 38), True, pts, 1)
                sx, sy, sz = project_3d(*torus_to_3d(th, 0))
                if sz > -50:
                    screen.blit(font.render(TORUS_FIFTHS[k], True, (0, 200, 255)), (sx-6, sy-6))

            for m in range(4):
                ph = m * (2.0 * math.pi / 4.0)
                pts = [project_3d(*torus_to_3d(t, ph))[:2] for t in np.linspace(0, 2.0*math.pi, 40)]
                pygame.draw.lines(screen, (30, 42, 56), True, pts, 1)

            # 2. Torus Avatar Trail
            if len(torus_trail) > 1:
                t_pts = [project_3d(*torus_to_3d(th, ph))[:2] for th, ph in torus_trail]
                pygame.draw.lines(screen, (0, 180, 255), False, t_pts, 2)

            # 3. Avatar Representation
            asx, asy, _ = project_3d(*torus_to_3d(theta_avatar, phi_avatar))
            hue = int(np.clip((ax.ry + 1.0) / 2.0 * 255, 0, 255))
            pygame.draw.circle(screen, (hue, 255 - hue, 255), (asx, asy), 11)
            pygame.draw.circle(screen, (255, 255, 255), (asx, asy), 4)

            # 4. Trajectory Vector Prediction (traj_32)
            base_gx, base_gy = node.traj_32[0, 0], node.traj_32[0, 1]
            traj_pts = [(asx, asy)]
            for k in range(1, NUM_FREQS):
                d_th = (node.traj_32[k, 0] - base_gx) * 0.4
                d_ph = (node.traj_32[k, 1] - base_gy) * 0.4
                psx, psy, _ = project_3d(*torus_to_3d(theta_avatar + d_th, phi_avatar - d_ph))
                traj_pts.append((psx, psy))

            for k in range(len(traj_pts) - 1):
                c_val = int(255 * (k / len(traj_pts)))
                pygame.draw.line(screen, (c_val, 140, 255 - c_val), traj_pts[k], traj_pts[k+1], 3)
            pygame.draw.circle(screen, (255, 50, 200), traj_pts[-1], 5)

            # 5. 26-mm Sensor Display (16 Electrodes, Active iPLV Edges)
            cx_p, cy_p, r_p = 770, 240, 110
            pygame.draw.circle(screen, (14, 20, 28), (cx_p, cy_p), r_p)
            pygame.draw.circle(screen, (0, 200, 255), (cx_p, cy_p), r_p, 1)
            
            sc = (r_p - 15) / 10.14
            px = cx_p + (COORDS_X * sc).astype(int)
            py = cy_p + (COORDS_Y * sc).astype(int)
            
            mean_iplv = np.mean(np.abs(node.iplv_32), axis=0)
            max_v = np.max(mean_iplv) + 1e-6
            for p in range(NUM_PAIRS):
                v = mean_iplv[p] / max_v
                if v > 0.15:
                    c = int(np.clip(v * 255, 50, 255))
                    pygame.draw.line(screen, (0, c, int(c*0.8)), (px[I_IDX[p]], py[I_IDX[p]]), (px[J_IDX[p]], py[J_IDX[p]]), 1)
            for c_i in range(16):
                pygame.draw.circle(screen, (255, 220, 0), (px[c_i], py[c_i]), 4)
                
            screen.blit(font.render("26-MM SENSOR ARRAY (120 iPLV)", True, (255, 255, 255)), (680, 100))

            # ------------------------------------------------------------------
            # UI CONTROL DASHBOARD
            # ------------------------------------------------------------------
            ui_y = HEIGHT - 140
            pygame.draw.rect(screen, (12, 16, 24), (0, ui_y, WIDTH, 140))
            pygame.draw.line(screen, (0, 200, 255), (0, ui_y), (WIDTH, ui_y), 2)

            k_curr = int(round((theta_avatar / (2.0 * math.pi)) * 12.0)) % 12
            k_next = (k_curr + 1) % 12
            blend_pct = ((theta_avatar / (2.0 * math.pi)) * 12.0) % 1.0

            # Gradient Morph Bar
            bar_x, bar_y, bar_w, bar_h = 580, ui_y + 15, 360, 14
            pygame.draw.rect(screen, (20, 30, 45), (bar_x, bar_y, bar_w, bar_h))
            fill_w = int(current_morph * bar_w)
            m_col = (int(current_morph * 255), int(180 * (1 - current_morph)), int(255 * (1 - current_morph)))
            pygame.draw.rect(screen, m_col, (bar_x, bar_y, fill_w, bar_h))
            pygame.draw.rect(screen, (0, 200, 255), (bar_x, bar_y, bar_w, bar_h), 1)

            stage_name = "DARK PSY-CHILL" if current_morph < 0.35 else ("PSY-PROGRESSIVE" if current_morph < 0.70 else "FULL-ON PSYTRANCE")
            screen.blit(font_lg.render(f"TONAL TORUS | {TORUS_FIFTHS[k_curr]} -> {TORUS_FIFTHS[k_next]} ({blend_pct*100:.0f}%)", True, (255, 255, 255)), (20, ui_y + 10))
            screen.blit(font.render(f"DANCEFLOOR [W/S]: {stage_name} | 120-EDGE MODAL PERCUSSION", True, m_col), (bar_x, ui_y + 35))
            
            sync_str = f"SYNC: {frame.theta_sync*100:.0f}%" + (f" | DEVICES: {frame.num_live}" if frame.num_live > 1 else "")
            screen.blit(font.render(f"NAVIGATION [lx, ly]: X={force_x:+.2f} | Y={force_y:+.2f} | {sync_str}", True, (0, 200, 255)), (20, ui_y + 35))
            
            rx_str = "STABLE (Consonant)" if abs(ax.rx) < 0.15 else f"TABLA/ZAP BEND (Acid +{abs(ax.rx):.2f})"
            screen.blit(font.render(f"SAGITTA [rx]: {ax.rx:+.2f} -> {rx_str}", True, (255, 200, 100)), (20, ui_y + 60))
            
            ry_str = "ROLLING BASS GALLOP (Future)" if ax.ry > 0.15 else ("HEAVY SUB-BODY (Past)" if ax.ry < -0.15 else "BALANCED")
            screen.blit(font.render(f"TEMPORAL BIAS [ry]: {ax.ry:+.2f} -> {ry_str} (Velocity x{accel_mult:.2f})", True, (255, 120, 220)), (20, ui_y + 85))

            pygame.display.flip()

    finally:
        stop_event.set()
        t_audio.join()
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
