#!/usr/bin/env python3
"""
🧠 NEUROCANVAS MAZE APP: Full 4-Axis Theta-Gamma Decomposition
Использует универсальные оси gamepad_axes из ядра движка.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import sys
import math
import random
import numpy as np
import pygame
from neuro_heterarchy_core import HeterarchicalBrainEngine, NUM_FREQS

DIM = 13
CELL_SIZE = 48
MAZE_W = DIM * CELL_SIZE
MAZE_H = DIM * CELL_SIZE

FIXED_WORLD_VIEW = True

class TopoMaze:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.grid = [[1 for _ in range(dim)] for _ in range(dim)]
        self._gen(1, 1)
        self.exit_pos = (dim - 2, dim - 2)
        self.grid[dim - 2][dim - 2] = 2
        self.bake_surface()

    def bake_surface(self):
        self.baked_surface = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
        self.baked_surface.fill((0, 0, 0, 0))
        for gy in range(self.dim):
            for gx in range(self.dim):
                if self.grid[gy][gx] == 1:
                    rect = (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.baked_surface, (16, 24, 34), rect)
                    pygame.draw.rect(self.baked_surface, (30, 48, 68), rect, 1)
                elif self.grid[gy][gx] == 2:
                    cx = int((gx + 0.5) * CELL_SIZE)
                    cy = int((gy + 0.5) * CELL_SIZE)
                    pygame.draw.circle(self.baked_surface, (0, 255, 120), (cx, cy), 16)

    def _gen(self, x, y):
        self.grid[y][x] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.dim - 1 and 0 < ny < self.dim - 1 and self.grid[ny][nx] == 1:
                self.grid[y + dy // 2][x + dx // 2] = 0
                self._gen(nx, ny)

    def is_wall(self, gx, gy):
        if gx < 0 or gx >= self.dim or gy < 0 or gy >= self.dim: return True
        return self.grid[int(gy)][int(gx)] == 1

def world_to_screen(x, y, avatar, fixed_view, offset_x=0, offset_y=0):
    if fixed_view:
        return offset_x + x * CELL_SIZE, offset_y + y * CELL_SIZE
    else:
        cx = offset_x + MAZE_W / 2.0
        cy = offset_y + MAZE_H / 2.0
        dx = (x - avatar.x) * CELL_SIZE
        dy = (y - avatar.y) * CELL_SIZE
        cam_angle = -avatar.angle - math.pi / 2.0
        rx = dx * math.cos(cam_angle) - dy * math.sin(cam_angle)
        ry = dx * math.sin(cam_angle) + dy * math.cos(cam_angle)
        return cx + rx, cy + ry

class DynamicAvatar:
    def __init__(self):
        self.x, self.y = 1.5, 1.5
        self.vx, self.vy = 0.0, 0.0
        self.angle = -math.pi / 2.0
        
        self.persistence = 0.0
        self.last_ix, self.last_iy = 0.0, 0.0
        
        self.ang_v = 0.0
        self.wm_turn_curve = 0.0     
        self.temporal_bias = 0.0     
        self.trail = []

    def update(self, dt, force_x, force_y, wm_curvature, temp_bias, fixed_view):
        self.wm_turn_curve = wm_curvature
        self.temporal_bias = temp_bias
        
        if not fixed_view:
            if abs(wm_curvature) > 0.08:
                target_ang_v = wm_curvature * 2.8
            else:
                target_ang_v = 0.0
            self.ang_v = self.ang_v * 0.82 + target_ang_v * 0.18
            self.angle += self.ang_v * dt
            self.angle = (self.angle + math.pi) % (2.0 * math.pi) - math.pi
        else:
            self.ang_v = 0.0

        mag = math.hypot(force_x, force_y)
        if mag > 0.05:
            last_mag = math.hypot(self.last_ix, self.last_iy) + 1e-6
            dot = (force_x * self.last_ix + force_y * self.last_iy) / (mag * last_mag)
            alignment = max(0.0, dot)
        else:
            alignment = 0.0

        self.persistence = self.persistence * 0.95 + 0.05 * alignment * math.tanh(mag * 2.0)
        self.last_ix, self.last_iy = force_x, force_y
        
        accel_multiplier = 1.0 + temp_bias
        active_boost = 1.0 + self.persistence * 4.0 
        base_speed = 3.2 * active_boost * accel_multiplier

        if fixed_view:
            target_vx = force_x * base_speed
            target_vy = force_y * base_speed
        else:
            forward_drive = -force_y
            lateral_strafe = force_x
            fwd_x, fwd_y = math.cos(self.angle), math.sin(self.angle)
            str_x, str_y = -math.sin(self.angle), math.cos(self.angle)
            target_vx = (fwd_x * forward_drive + str_x * lateral_strafe) * base_speed
            target_vy = (fwd_y * forward_drive + str_y * lateral_strafe) * base_speed
        
        MAX_SPEED = 9.0
        t_mag = math.hypot(target_vx, target_vy)
        if t_mag > MAX_SPEED:
            target_vx = (target_vx / t_mag) * MAX_SPEED
            target_vy = (target_vy / t_mag) * MAX_SPEED
        
        self.vx = self.vx * 0.88 + target_vx * 0.12
        self.vy = self.vy * 0.88 + target_vy * 0.12

    def move(self, dt, maze):
        move_dist = math.hypot(self.vx, self.vy) * dt
        steps = max(1, int(math.ceil(move_dist / 0.04)))
        sdx = (self.vx * dt) / steps
        sdy = (self.vy * dt) / steps
        r = 0.22 

        for _ in range(steps):
            if not maze.is_wall(self.x + sdx + math.copysign(r, sdx), self.y): 
                self.x += sdx
            if not maze.is_wall(self.x, self.y + sdy + math.copysign(r, sdy)): 
                self.y += sdy

        if int(self.x) == maze.exit_pos[0] and int(self.y) == maze.exit_pos[1]:
            maze.__init__(DIM)
            self.x, self.y = 1.5, 1.5

        self.trail.append((self.x, self.y))
        if len(self.trail) > 35: self.trail.pop(0)

def main():
    global FIXED_WORLD_VIEW
    import multiprocessing as mp
    mp.freeze_support()

    pygame.init()
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((1260, 720), flags, vsync=0)
    pygame.display.set_caption("NeuroCanvas: 4-Axis Theta-Gamma Engine (Sagitta & Temporal Bias)")
    clock = pygame.time.Clock()

    engine = HeterarchicalBrainEngine()
    engine.start()

    maze = TopoMaze(DIM)
    avatar = DynamicAvatar()
    
    N_elements = 2      
    lead_idx = 0   
    sample_indices = np.linspace(0, NUM_FREQS - 1, N_elements, dtype=int)

    try:
        while True:
            dt = clock.tick(0) / 1000.0
            dt = min(0.05, dt)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if pygame.K_1 <= event.key <= pygame.K_4: lead_idx = event.key - pygame.K_1
                    elif pygame.K_F5 <= event.key <= pygame.K_F8: lead_idx = event.key - pygame.K_F5
                    elif event.key in (pygame.K_F1, pygame.K_MINUS, pygame.K_q): 
                        N_elements = max(2, N_elements - 1)
                        sample_indices = np.linspace(0, NUM_FREQS - 1, N_elements, dtype=int)
                    elif event.key in (pygame.K_F2, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_e): 
                        N_elements = min(NUM_FREQS, N_elements + 1)
                        sample_indices = np.linspace(0, NUM_FREQS - 1, N_elements, dtype=int)
                    elif event.key in (pygame.K_F3, pygame.K_TAB, pygame.K_v): 
                        FIXED_WORLD_VIEW = not FIXED_WORLD_VIEW
                    elif event.key in (pygame.K_F4, pygame.K_0, pygame.K_r): 
                        maze.__init__(DIM)
                        avatar.x, avatar.y = 1.5, 1.5

            frame = engine.get_frame()
            nodes = [frame.fcz_macro, frame.pz_spatial, frame.oz_sensory, frame.cz_motor]
            lead_node = nodes[lead_idx]

            # -------------------------------------------------------------
            # НОВЫЙ ДОСТУП К 4 ОСЯМ ИЗ ЯДРА (Никакой ручной математики)
            # -------------------------------------------------------------
            axes = lead_node.gamepad_axes
            
            # В ядре мы делали инверсию (-ly) для геймпада (где Вперед = +). 
            # Для экрана Pygame мы возвращаем оригинальный знак, делая еще один минус (-axes.ly)
            force_x = axes.lx
            force_y = -axes.ly
            wm_curvature = axes.rx
            temporal_bias = axes.ry

            avatar.update(dt, force_x, force_y, wm_curvature, temporal_bias, FIXED_WORLD_VIEW)
            avatar.move(dt, maze)

            # ==================================================================
            # РЕНДЕРИНГ
            # ==================================================================
            screen.fill((6, 8, 12))
            ox, oy = 30, 45
            
            maze_surface = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
            maze_surface.fill((0, 0, 0, 0))

            cam_angle = -avatar.angle - math.pi / 2.0
            cos_c = math.cos(cam_angle)
            sin_c = math.sin(cam_angle)

            if FIXED_WORLD_VIEW:
                maze_surface.blit(maze.baked_surface, (0, 0))
                px = int(avatar.x * CELL_SIZE)
                py = int(avatar.y * CELL_SIZE)
                draw_ang = avatar.angle
            else:
                rot_deg = math.degrees(-cam_angle)
                rot_surf = pygame.transform.rotate(maze.baked_surface, rot_deg)
                
                ax = avatar.x * CELL_SIZE
                ay = avatar.y * CELL_SIZE
                vx_c = (MAZE_W / 2.0) - ax
                vy_c = (MAZE_H / 2.0) - ay
                
                rx_c = vx_c * cos_c - vy_c * sin_c
                ry_c = vx_c * sin_c + vy_c * cos_c
                
                blit_x = int((MAZE_W / 2.0 + rx_c) - rot_surf.get_width() / 2.0)
                blit_y = int((MAZE_H / 2.0 + ry_c) - rot_surf.get_height() / 2.0)
                maze_surface.blit(rot_surf, (blit_x, blit_y))
                
                px = int(MAZE_W / 2.0)
                py = int(MAZE_H / 2.0)
                draw_ang = -math.pi / 2.0

            if len(avatar.trail) > 1:
                trail_pts = [world_to_screen(tx, ty, avatar, FIXED_WORLD_VIEW, 0, 0) for tx, ty in avatar.trail]
                pygame.draw.lines(maze_surface, (0, 180, 255), False, trail_pts, 2)

            if not FIXED_WORLD_VIEW and abs(avatar.ang_v) > 0.10:
                arc_r = 28
                sign = 1 if avatar.ang_v > 0 else -1
                span_rad = min(math.pi * 0.7, abs(avatar.ang_v) * 0.5)
                
                arc_pts = []
                for a_step in np.linspace(0, span_rad, 16):
                    th = draw_ang + sign * a_step
                    arc_pts.append((px + int(arc_r * math.cos(th)), py + int(arc_r * math.sin(th))))
                    
                if len(arc_pts) > 1:
                    col_arc = (255, 50, 200) if sign > 0 else (0, 200, 255)
                    pygame.draw.lines(maze_surface, col_arc, False, arc_pts, 4)
                    pygame.draw.circle(maze_surface, (255, 255, 255), arc_pts[-1], 4)

            VEC_SCALE = 120.0
            
            if lead_idx == 3:
                ev_x = px + force_x * VEC_SCALE
                ev_y = py + force_y * VEC_SCALE
                pygame.draw.line(maze_surface, (255, 220, 0), (px, py), (int(ev_x), int(ev_y)), 4)
                pygame.draw.circle(maze_surface, (255, 255, 255), (int(ev_x), int(ev_y)), 4)
            else:
                spline_pts = [(px, py)]
                base_gx = lead_node.traj_32[0, 0]
                base_gy = lead_node.traj_32[0, 1]
                for idx_k in sample_indices[1:]:
                    sk_x = lead_node.traj_32[idx_k, 0] - base_gx
                    sk_y = lead_node.traj_32[idx_k, 1] - base_gy
                    sk_len = math.hypot(sk_x, sk_y)
                    if sk_len > 1.0: sk_x /= sk_len; sk_y /= sk_len
                    
                    sp_x = px + sk_x * VEC_SCALE
                    sp_y = py + sk_y * VEC_SCALE
                    spline_pts.append((int(sp_x), int(sp_y)))

                if N_elements == 2:
                    pygame.draw.line(maze_surface, (0, 255, 255), spline_pts[0], spline_pts[1], 5)
                    pygame.draw.circle(maze_surface, (255, 255, 255), spline_pts[1], 5)
                else:
                    for k in range(len(spline_pts) - 1):
                        c_val = int(255 * (k / max(1, len(spline_pts) - 1)))
                        col = (c_val, 140, 255 - c_val)
                        pygame.draw.line(maze_surface, col, spline_pts[k], spline_pts[k + 1], 4)
                        pygame.draw.circle(maze_surface, (255, 255, 255), spline_pts[k + 1], 3)
                    pygame.draw.circle(maze_surface, (255, 100, 255), spline_pts[-1], 6)

            accel_hue = int(np.clip((avatar.temporal_bias + 1.0) / 2.0 * 255, 0, 255))
            pygame.draw.circle(maze_surface, (accel_hue, 255 - accel_hue, 255), (px, py), 12)
            nx = px + int(math.cos(draw_ang) * 20)
            ny = py + int(math.sin(draw_ang) * 20)
            pygame.draw.line(maze_surface, (255, 255, 255), (px, py), (nx, ny), 3)

            screen.blit(maze_surface, (ox, oy))
            pygame.draw.rect(screen, (0, 200, 255), (ox, oy, MAZE_W, MAZE_H), 2)

            rx = 680
            colors = [(255, 50, 200), (0, 200, 255), (100, 255, 100), (255, 180, 0)]
            f_b = pygame.font.SysFont("consolas", 13, bold=True)
            f_s = pygame.font.SysFont("consolas", 11)
            
            pygame.draw.rect(screen, (12, 16, 22), (rx, 45, 540, 105))
            pygame.draw.rect(screen, (0, 255, 200), (rx, 45, 540, 105), 2)
            
            cam_mode = "WORLD [Fixed]" if FIXED_WORLD_VIEW else "3RD PERSON [Ego]"
            screen.blit(f_b.render(f"КАМЕРА [F3/V]: {cam_mode} | ЯДРО: 4-AXIS THETA-GAMMA", True, (0, 255, 200)), (rx + 12, 53))
            
            pts_label = "2 ЭЛЕМЕНТА (1 СЕГМЕНТ - РЕФЛЕКС)" if N_elements == 2 else f"{N_elements} ЭЛЕМЕНТОВ (МУЛЬТИ-ШАГ)"
            screen.blit(f_s.render(f"РАЗРЕШЕНИЕ [F1/F2]: {pts_label}", True, (255, 220, 0)), (rx + 12, 75))
            
            bias = avatar.temporal_bias
            if bias > 0.15:
                tb_str = f">>> БУДУЩЕЕ (PITCH UP / TURBO): {bias:+.2f}"
                tb_col = (255, 50, 200)
            elif bias < -0.15:
                tb_str = f"<<< ПРОШЛОЕ (PITCH DOWN / BRAKE): {bias:+.2f}"
                tb_col = (0, 200, 255)
            else:
                tb_str = "0.00 (БАЛАНС / СТАБИЛЬНЫЙ ПОЛЕТ)"
                tb_col = (0, 255, 120)
                
            screen.blit(f_b.render(f"ТЕМПОРАЛЬНЫЙ СДВИГ: {tb_str}", True, tb_col), (rx + 12, 97))

            for i, n in enumerate(nodes):
                ry_ui = 160 + i * 115
                col = colors[i]
                is_lead = (i == lead_idx)
                
                pygame.draw.rect(screen, (12, 16, 22), (rx, ry_ui, 540, 105))
                pygame.draw.rect(screen, col if is_lead else (50, 60, 70), (rx, ry_ui, 540, 105), 2 if is_lead else 1)
                
                role_label = f"[{i+1}/F{i+5}] {n.name}" + (" [ВЕДУЩИЙ ◄]" if is_lead else "")
                screen.blit(f_b.render(role_label, True, col if is_lead else (130, 140, 150)), (rx + 12, ry_ui + 10))
                screen.blit(f_s.render(f"Вихрь Tq: {n.tq:+.2f} | Тяга: {n.thrust:.2f}", True, (160, 170, 180)), (rx + 12, ry_ui + 32))

                cx_radar, cy_radar, rc = rx + 475, ry_ui + 52, 38
                pygame.draw.circle(screen, (20, 28, 38), (cx_radar, cy_radar), rc, 1)
                pygame.draw.line(screen, (30, 40, 50), (cx_radar - rc, cy_radar), (cx_radar + rc, cy_radar), 1)
                pygame.draw.line(screen, (30, 40, 50), (cx_radar, cy_radar - rc), (cx_radar, cy_radar + rc), 1)

                nbx, nby = n.vx, n.vy
                nb_len = math.hypot(nbx, nby)
                if nb_len > 1.0: nbx /= nb_len; nby /= nb_len
                pygame.draw.line(screen, (255, 220, 0), (cx_radar, cy_radar), (cx_radar + int(nbx * (rc - 4)), cy_radar + int(nby * (rc - 4))), 2)

                pts_r = [(cx_radar, cy_radar)]
                base_gx_r = n.traj_32[0, 0]
                base_gy_r = n.traj_32[0, 1]
                for idx_k in sample_indices[1:]:
                    rx_k = n.traj_32[idx_k, 0] - base_gx_r
                    ry_k = n.traj_32[idx_k, 1] - base_gy_r
                    rk_len = math.hypot(rx_k, ry_k)
                    if rk_len > 1.0: rx_k /= rk_len; ry_k /= rk_len
                    pts_r.append((cx_radar + int(rx_k * (rc - 4)), cy_radar + int(ry_k * (rc - 4))))

                for k in range(len(pts_r) - 1):
                    c_r = int(255 * (k / max(1, len(pts_r) - 1)))
                    pygame.draw.line(screen, (c_r, 120, 255 - c_r), pts_r[k], pts_r[k + 1], 2)
                if len(pts_r) > 1:
                    pygame.draw.circle(screen, col, pts_r[-1], 4)

            fps_val = clock.get_fps()
            h_stat = pygame.font.SysFont("consolas", 14, bold=True)
            screen.blit(h_stat.render(f"NEUROCANVAS CORE | LIVE EEG | {fps_val:.0f} FPS", True, (255, 255, 255)), (30, 15))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        pygame.quit()

if __name__ == '__main__':
    main()
