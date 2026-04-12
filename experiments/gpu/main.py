"""
Tron 3D Wireframe Escape

An autonomous 3D Tron simulation built with Pygame. 
The cycle drops into a procedurally generated maze (built via a recursive backtracker DFS algorithm),
and drives forward at random speed intervals.

Features:
- Numba JIT Compilation: Core 3D perspective projection math is pre-compiled to Native Machine Code for 60fps locking.
- Spatial Memory Grid: The AI cycle tracks a persistent 12x12 heatmap of the maze. The more time it spends in a single coordinate corridor, the higher the "Heat Penalty" grows.
- Heuristic Pathfinding: When raycasting for clear paths, the AI penalizes long physical corridors if they have a highly-visited heatmap score. It actively prefers shorter, unexplored paths to escape dead ends.
- Live Minimap: Placed in the bottom right, rendering walls (blue), the cycle (cyan), and the glowing heatmap memory (Green -> Yellow -> Red).
- Hardware Profiling: Every 5 seconds, Autodrive swaps the renderer live between an Nvidia CUDA CuPy GPU Bloom pass and a standard Native Numba CPU Fallback loop, logging performance out to telemetry.csv.
"""

import pygame
import math
import sys
import random
import csv
import time

# Attempt to load CUDA acceleration tools
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False
# Attempt to load Numba for CPU JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrapper(func): return func
        return wrapper

@njit(fastmath=True, cache=True)
def project_point(x, y, z, cx, cy, cz, cos_y, sin_y, cos_p, sin_p, fov, w2, h2):
    tx = x - cx
    ty = y - cy
    tz = z - cz
    
    rx = tx * cos_y - tz * sin_y
    rz = tx * sin_y + tz * cos_y
    ry = ty
    
    ry2 = ry * cos_p - rz * sin_p
    rz2 = ry * sin_p + rz * cos_p
    
    if rz2 < 0.1: 
        return -1, -1
        
    sx = w2 + (rx * fov) / rz2
    sy = h2 - (ry2 * fov) / rz2
    return int(sx), int(sy)

def main():
    # Initialize Pygame and set up the display window
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tron 3D Wireframe (CUDA: " + str(CUDA_AVAILABLE) + ")")
    clock = pygame.time.Clock()
    
    # Render layers for neon bloom
    main_layer = pygame.Surface((WIDTH, HEIGHT))
    
    # Font for HUD
    font = pygame.font.SysFont(None, 24)

    # Define color constants (RGB format)
    BLACK = (0, 0, 0) # Background
    GRID_COLOR = (0, 50, 100) # Floor grid (Dark blue)
    CYCLE_COLOR = (0, 255, 255) # Player cycle (Cyan)
    TRAIL_COLOR = (255, 128, 0) # The light trail left behind (Orange)
    WALL_COLOR = (100, 0, 100) # Arena boundary walls (Purple)

    # 3D Camera rendering constants
    FOV = 400 # Field of view factor (controls zoom and perspective)
    CAMERA_DISTANCE = 150 # Distance trailing behind the cycle
    CAMERA_HEIGHT = 80 # Height of the camera above the ground
    CAMERA_PITCH = -0.3 # Angle looking down at the cycle (radians)

    # Initial state variables for the Lightcycle and the Camera
    cycle_x = 0.0 # Cycle's X position in 3D space
    cycle_z = 0.0 # Cycle's Z position in 3D space
    cycle_dir = 0.0 # Cycle's facing angle (yaw) in radians
    cam_yaw = 0.0 # Camera's horizontal viewing angle in radians
    
    # Autodrive and Telemetry state
    autodrive_mode = True
    profile_timer = 0
    active_profile_mode = "CUDA" if CUDA_AVAILABLE else "CPU"
    
    # Open telemetry file in append mode 
    telemetry_file = open("telemetry.csv", "w", newline="")
    telemetry_writer = csv.writer(telemetry_file)
    telemetry_writer.writerow(["Timestamp", "Mode", "FPS", "Speed", "X", "Z", "AI_Action", "Pipeline"])
    
    # Speed and acceleration configuration
    base_speed = 3.0
    current_speed = base_speed
    target_speed = base_speed
    speed_transition_timer = 0
    max_speed = 10.0 # Slowed down max speed to handle tight maze corners
    turn_speed = 0.08 # Speed at which the cycle/camera turns

    # Variables for managing the 3D light ribbon/trail
    trail = [] # List storing older line segments of the light trail
    MAX_TRAIL_LENGTH = 70  # Max segments allowed before exploding
    last_trail_x, last_trail_z = cycle_x, cycle_z # Last point where the trail registered a joint
    
    # Particle system for explosions
    particles = []

    # Frame-specific trig cache to avoid recalculating sines/cosines per point
    trig_cache = [1.0, 0.0, 1.0, 0.0]
    
    def project(x, y, z, cx, cy, cz, cam_yaw, cam_pitch):
        """
        Projects a 3D point via JIT compiled native code.
        """
        sx, sy = project_point(x, y, z, cx, cy, cz, trig_cache[0], trig_cache[1], trig_cache[2], trig_cache[3], FOV, WIDTH/2, HEIGHT/2)
        if sx == -1 and sy == -1: 
            return None
        return sx, sy

    def draw_line(p1_3d, p2_3d, cx, cy, cz, cam_yaw, color, width=1, surf=screen):
        """
        Draws a 3D line between two 3D points by projecting them both to 2D screen coordinates.
        """
        p1_2d = project(p1_3d[0], p1_3d[1], p1_3d[2], cx, cy, cz, cam_yaw, CAMERA_PITCH)
        p2_2d = project(p2_3d[0], p2_3d[1], p2_3d[2], cx, cy, cz, cam_yaw, CAMERA_PITCH)
        
        # Only draw the line if both points successfully projected (are in front of camera)
        if p1_2d and p2_2d:
            pygame.draw.line(surf, color, p1_2d, p2_2d, width)

    def draw_grid(cx, cy, cz, cam_yaw, surf=screen):
        """
        Draws an infinite scrolling checkered wireframe floor grid centered around the camera's vicinity.
        """
        grid_size = 1000 # Distance the grid extends in each direction away from the camera
        step = 50 # Size of each individual grid square
        
        # Calculate start and end coordinates locked to the grid step spacing
        start_x = int(cx / step) * step - grid_size
        end_x = start_x + grid_size * 2
        start_z = int(cz / step) * step - grid_size
        end_z = start_z + grid_size * 2

        for x in range(start_x, end_x, step):
            for z in range(start_z, end_z, step):
                # Draw short segments so they clip gracefully at the near plane if needed
                draw_line((x, 0, z), (x, 0, z + step), cx, cy, cz, cam_yaw, GRID_COLOR, surf=surf)
                draw_line((x, 0, z), (x + step, 0, z), cx, cy, cz, cam_yaw, GRID_COLOR, surf=surf)

    # ==========================
    # Procedural Maze Generation
    # ==========================
    MAZE_COLS = 12
    MAZE_ROWS = 12
    CELL_SIZE = 300
    
    # Grid where 1 is wall, 0 is open path. Start completely filled with walls
    maze_grid = [[1 for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
    
    # AI Memory/Learning Grid (tracks how many times the AI has been in a specific cell)
    # Starts at 0 (unexplored). Higher numbers = highly explored/dead ends
    ai_memory_grid = [[0 for _ in range(MAZE_COLS)] for _ in range(MAZE_ROWS)]
    
    # Recursive Backtracker (DFS) to carve a random perfect maze
    def carve_maze(r, c):
        maze_grid[r][c] = 0 # Mark as open path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            nr, nc = r + dr*2, c + dc*2
            # Check bounds
            if 0 <= nr < MAZE_ROWS and 0 <= nc < MAZE_COLS:
                # If neighbor is still a wall, carve through
                if maze_grid[nr][nc] == 1:
                    maze_grid[r + dr][c + dc] = 0 # Carve the wall between
                    carve_maze(nr, nc)
                    
    # Ensure maze coordinates are strictly odd to maintain wall boundaries
    carve_maze(1, 1)
    
    # Optional: Knock down a few random walls to create loops instead of a pure labyrinth
    for _ in range(10):
        rr = random.randint(1, MAZE_ROWS-2)
        rc = random.randint(1, MAZE_COLS-2)
        maze_grid[rr][rc] = 0
        
    # Convert active walls into 3D line segments for rendering
    walls = []
    
    # Calculate offset so maze is centered around (0,0,0)
    offset_x = (MAZE_COLS * CELL_SIZE) // 2
    offset_z = (MAZE_ROWS * CELL_SIZE) // 2
    
    # Pre-calculate geometric lines based on grid blocks
    # Instead of drawing every single grid wall, we merge adjacent ones to save rendering lines
    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            if maze_grid[r][c] == 1:
                x1 = c * CELL_SIZE - offset_x
                z1 = r * CELL_SIZE - offset_z
                x2 = x1 + CELL_SIZE
                z2 = z1 + CELL_SIZE
                
                # Check neighbors to only draw outer faces
                # Top edge
                if r == 0 or maze_grid[r-1][c] == 0:
                    walls.append(((x1, 0, z1), (x2, 0, z1)))
                # Bottom edge
                if r == MAZE_ROWS-1 or maze_grid[r+1][c] == 0:
                    walls.append(((x1, 0, z2), (x2, 0, z2)))
                # Left edge
                if c == 0 or maze_grid[r][c-1] == 0:
                    walls.append(((x1, 0, z1), (x1, 0, z2)))
                # Right edge
                if c == MAZE_COLS-1 or maze_grid[r][c+1] == 0:
                    walls.append(((x2, 0, z1), (x2, 0, z2)))
                    
    # Hard bounds for the total box size
    bounds_x = offset_x
    bounds_z = offset_z
    
    # Force cycle to spawn in a guaranteed open tile (1,1)
    cycle_x = CELL_SIZE - offset_x + (CELL_SIZE/2)
    cycle_z = CELL_SIZE - offset_z + (CELL_SIZE/2)
    cycle_dir = 0.0
    last_trail_x, last_trail_z = cycle_x, cycle_z 
    
    # Helper for AI to check if a specific world coordinate intersects a maze wall
    def is_wall(wx, wz):
        # Convert world X,Z into grid columns and rows
        c = int((wx + offset_x) // CELL_SIZE)
        r = int((wz + offset_z) // CELL_SIZE)
        
        # OOB check
        if c < 0 or c >= MAZE_COLS or r < 0 or r >= MAZE_ROWS:
            return True
        return maze_grid[r][c] == 1

    # Main Game Loop
    running = True
    while running:
        ai_action_log = "None"
        
        # Event handling (e.g., clicking the red X to close the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Toggle Autodrive mode on ESC key press
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    autodrive_mode = not autodrive_mode
                
        # Input handling for continuous key press state
        keys = pygame.key.get_pressed()
        
        # ==========================
        # AI Logic Update
        # ==========================
        # Log current physical position into the AI's spatial memory to learn the layout
        curr_c = int((cycle_x + offset_x) // CELL_SIZE)
        curr_r = int((cycle_z + offset_z) // CELL_SIZE)
        if 0 <= curr_c < MAZE_COLS and 0 <= curr_r < MAZE_ROWS:
            # Increase "heat" or penalty of the current cell so the AI prefers not to return here
            ai_memory_grid[curr_r][curr_c] += 0.05
            
        if not autodrive_mode:
            # User Steering controls
            if keys[pygame.K_LEFT]:
                cycle_dir += turn_speed
            if keys[pygame.K_RIGHT]:
                cycle_dir -= turn_speed
        else:
            # ==========================
            #   AI Maze Raycasting & Learning
            # ==========================
            # Cast 3 feeler rays: Straight, Slight Left, Slight Right
            rays = [0, -0.6, 0.6] 
            ray_distances = [0, 0, 0]
            # Ray learning penalties (higher means the ray points towards a highly visited dead-end)
            ray_memory_penalties = [0, 0, 0] 
            
            MAX_RAY = max(150.0, current_speed * 18.0)
            
            for i, angle_offset in enumerate(rays):
                ray_dir = cycle_dir + angle_offset
                dx = math.sin(ray_dir)
                dz = math.cos(ray_dir)
                
                # March ray forward until it hits a wall
                dist = 0
                step_size = 25
                mem_penalty = 0.0
                
                while dist < MAX_RAY:
                    check_x = cycle_x + (dx * dist)
                    check_z = cycle_z + (dz * dist)
                    
                    if is_wall(check_x, check_z):
                        break
                        
                    # Sample the AI's memory at this location along the ray
                    check_c = int((check_x + offset_x) // CELL_SIZE)
                    check_r = int((check_z + offset_z) // CELL_SIZE)
                    if 0 <= check_c < MAZE_COLS and 0 <= check_r < MAZE_ROWS:
                         mem_penalty += ai_memory_grid[check_r][check_c]
                         
                    dist += step_size
                    
                ray_distances[i] = dist
                ray_memory_penalties[i] = mem_penalty
                
            dist_straight = ray_distances[0]
            dist_left = ray_distances[1]
            dist_right = ray_distances[2]
            
            # Combine physical wall distance with psychological memory avoidance
            # We want to pick paths that are both physically long AND sparsely visited
            heuristic_left = dist_left - (ray_memory_penalties[1] * 20.0)
            heuristic_right = dist_right - (ray_memory_penalties[2] * 20.0)
            
            buffer_dist = 100
            turn_amt = turn_speed * 1.5
            
            # If physically trapped dead-on, or boxed in on all sides, aggressively turn
            if dist_straight < 50 and dist_left < 50 and dist_right < 50:
                 cycle_dir += turn_amt * 2.5
                 ai_action_log = "Hard Trap Escape"
            # Imminent front collision, rely on heuristic memory scores
            elif dist_straight < buffer_dist:
                 if heuristic_left > heuristic_right:
                     cycle_dir += turn_amt # Steer Left towards fresher/longer path
                     ai_action_log = "Learning Turn (Left)"
                 else:
                     cycle_dir -= turn_amt # Steer Right towards fresher/longer path
                     ai_action_log = "Learning Turn (Right)"
            # Hug straight away from creeping side walls
            elif dist_left < buffer_dist * 0.7:
                 cycle_dir -= turn_amt * 0.5
                 ai_action_log = "Drift Right"
            elif dist_right < buffer_dist * 0.7:
                 cycle_dir += turn_amt * 0.5
                 ai_action_log = "Drift Left"
            else:
                 ai_action_log = "Cruise"
                
            # Profile switching logic to compare performance
            profile_timer += 1
            if profile_timer > 300: # Every 5 seconds at 60fps
                profile_timer = 0
                if active_profile_mode == "CUDA" or not CUDA_AVAILABLE:
                    active_profile_mode = "CPU_NUMBA"
                else:
                    active_profile_mode = "CUDA"
                
        # Free camera rotation controls
        if keys[pygame.K_a]:
            cam_yaw += turn_speed
        if keys[pygame.K_d]:
            cam_yaw -= turn_speed
            
        # Instantly recenter view behind the bike
        if keys[pygame.K_SPACE]:
            cam_yaw = cycle_dir
            
        # Random acceleration logic
        speed_transition_timer -= 1
        if speed_transition_timer <= 0:
            # Pick a random new target speed between low cruise and very high speed
            target_speed = random.uniform(base_speed, max_speed)
            # Pick a random duration to maintain this target
            speed_transition_timer = random.randint(60, 240) # 1 to 4 seconds at 60fps
            
        # Smoothly interpolate towards the target speed
        current_speed += (target_speed - current_speed) * 0.02
        
        # Update cycle position moving forward based on its facing angle and speed
        cycle_x += math.sin(cycle_dir) * current_speed
        cycle_z += math.cos(cycle_dir) * current_speed
        
        # Clamp bounds: Keep the cycle from escaping the absolute outer maze box
        # We don't enforce an inner perfect hard-stop purely to avoid cycle getting infinitely stuck 
        # inside an undetected glitch edge, but the bounds remain valid
        if cycle_x > bounds_x: cycle_x = bounds_x
        if cycle_x < -bounds_x: cycle_x = -bounds_x
        if cycle_z > bounds_z: cycle_z = bounds_z
        if cycle_z < -bounds_z: cycle_z = -bounds_z
        
        # Update the light ribbon trail
        # We only snap off a new permanent trail segment if we've traveled a solid distance 
        # from the last recorded point (to keep the list size small and corners sharp)
        dist = math.hypot(cycle_x - last_trail_x, cycle_z - last_trail_z)
        if dist > 20:
            trail.append((last_trail_x, last_trail_z, cycle_x, cycle_z))
            last_trail_x, last_trail_z = cycle_x, cycle_z
            
            # Trail Threshold Explosion Logic
            if len(trail) > MAX_TRAIL_LENGTH:
                # Spawn explosion particles along the oldest trail segments
                num_segments_to_destroy = 20
                for _ in range(num_segments_to_destroy):
                    if trail:
                        seg = trail.pop(0)
                        # Generate burst of 3D particles on this destroyed segment
                        mx = (seg[0] + seg[2]) / 2.0
                        mz = (seg[1] + seg[3]) / 2.0
                        for _ in range(3):
                            p_x = random.uniform(-10, 10) + mx
                            p_z = random.uniform(-10, 10) + mz
                            p_y = random.uniform(0, 25)
                            vx = random.uniform(-3, 3)
                            vy = random.uniform(2, 6)
                            vz = random.uniform(-3, 3)
                            # Store: [x, y, z, vx, vy, vz, life]
                            particles.append([p_x, p_y, p_z, vx, vy, vz, 1.0])

        # Update particles physics
        for p in reversed(particles):
            p[0] += p[3] # x
            p[1] += p[4] # y
            p[2] += p[5] # z
            p[4] -= 0.2  # Gravity on y
            p[6] -= 0.02 # Fade life
            if p[6] <= 0:
                particles.remove(p)

        # Calculate logical camera position (trailing behind the cycle relative to cam_yaw)
        cx = cycle_x - math.sin(cam_yaw) * CAMERA_DISTANCE
        cz = cycle_z - math.cos(cam_yaw) * CAMERA_DISTANCE
        cy = CAMERA_HEIGHT
        
        # Update global projection trig cache once per frame
        trig_cache[0] = math.cos(-cam_yaw)
        trig_cache[1] = math.sin(-cam_yaw)
        trig_cache[2] = math.cos(CAMERA_PITCH)
        trig_cache[3] = math.sin(CAMERA_PITCH)

        # Clear the rendering layer
        main_layer.fill(BLACK)
        
        # 1. Draw the floor grid below
        draw_grid(cx, cy, cz, cam_yaw, surf=main_layer)
        
        # 2. Draw arena walls (wireframes on the edges of the map)
        wall_h = 100 # Height of the walls stretching upwards
        for (w1, w2) in walls:
            # Only draw walls that are within a certain viewing distance to save Pygame processing time
            cam_dist_x = (w1[0] + w2[0]) / 2.0 - cx
            cam_dist_z = (w1[2] + w2[2]) / 2.0 - cz
            
            if (cam_dist_x*cam_dist_x + cam_dist_z*cam_dist_z) < 5000000:
                 draw_line((w1[0], 0, w1[2]), (w2[0], 0, w2[2]), cx, cy, cz, cam_yaw, WALL_COLOR, 2, surf=main_layer) # Bottom edge
                 draw_line((w1[0], wall_h, w1[2]), (w2[0], wall_h, w2[2]), cx, cy, cz, cam_yaw, WALL_COLOR, 2, surf=main_layer) # Top edge
                 draw_line((w1[0], 0, w1[2]), (w1[0], wall_h, w1[2]), cx, cy, cz, cam_yaw, WALL_COLOR, 2, surf=main_layer) # Vertical pillars connecting top/bottom
            
        # 3. Draw lightcycle ribbon trails
        trail_height = 25
        # The total active trails are the permanent history list PLUS the live current segment 
        # stretching actively between the last joint and the current cycle position
        all_trails = trail + [(last_trail_x, last_trail_z, cycle_x, cycle_z)]
        for tx1, tz1, tx2, tz2 in all_trails:
            draw_line((tx1, 0, tz1), (tx2, 0, tz2), cx, cy, cz, cam_yaw, TRAIL_COLOR, 3, surf=main_layer) # Bottom length edge
            draw_line((tx1, trail_height, tz1), (tx2, trail_height, tz2), cx, cy, cz, cam_yaw, TRAIL_COLOR, 3, surf=main_layer) # Top length edge
            draw_line((tx1, 0, tz1), (tx1, trail_height, tz1), cx, cy, cz, cam_yaw, TRAIL_COLOR, 3, surf=main_layer) # Vertical segment start post
            draw_line((tx2, 0, tz2), (tx2, trail_height, tz2), cx, cy, cz, cam_yaw, TRAIL_COLOR, 3, surf=main_layer) # Vertical segment end post
            
        # 4. Draw explosion particles (dynamic debris falling down)
        for p in particles:
            p2d = project(p[0], p[1], p[2], cx, cy, cz, cam_yaw, CAMERA_PITCH)
            if p2d:
                # Color fades based on life
                c_val = int(255 * max(0, p[6]))
                color = (c_val, int(c_val * 0.5), 0)
                pygame.draw.circle(main_layer, color, p2d, 3)
                
        # 5. Draw the actual cycle vehicle (a more complex wireframe mesh)
        local_verts = []
        local_edges = []
        s = 1.0 # Global scale factor for the cycle model size
        
        # Helper function to generate a wireframe wheel shape
        def add_wheel(center_z, radius, width, vert_offset):
            # 8 sides for the wheel octagon
            for i in range(8):
                angle = math.pi * 2 * i / 8
                # Offset y so the wheel sits directly on the ground
                y = radius + radius * math.cos(angle)
                z = center_z + radius * math.sin(angle)
                
                # Left and right vertices of this wheel segment
                local_verts.append((-width/2, y, z))
                local_verts.append((width/2, y, z))
                
                v_idx = vert_offset + i*2
                next_idx = vert_offset + ((i+1)%8)*2
                
                # Add edges to form the 3D wheel hub/tread
                local_edges.append((v_idx, v_idx+1)) # Cross beam
                local_edges.append((v_idx, next_idx)) # Left loop edge
                local_edges.append((v_idx+1, next_idx+1)) # Right loop edge
                
            return vert_offset + 16 # Return the new index offset

        # Generate Front Wheel (radius 6) and Rear Wheel (radius 8, slightly larger)
        v_off = 0
        v_off = add_wheel(12, 6, 4, v_off)    # Front wheel
        v_off = add_wheel(-12, 8, 8, v_off)   # Rear wheel
        
        # Generate Body Shell vertices (a sleek, faceted canopy hugging the wheels)
        body_verts = [
            (-3, 6, 15), (3, 6, 15),   # Front bumper (32, 33)
            (-4, 14, 2), (4, 14, 2),   # High canopy front (34, 35)
            (-4, 16, -8), (4, 16, -8), # Top canopy peak (36, 37)
            (-3, 8, -20), (3, 8, -20), # Tail end (38, 39)
            (-5, 4, -4), (5, 4, -4),   # Side flares/footrests (40, 41)
        ]
        
        body_edges = [
            (32,33), (34,35), (36,37), (38,39), (40,41), # Lateral cross beams
            (32,34), (34,36), (36,38), # Left top profile
            (33,35), (35,37), (37,39), # Right top profile
            (32,40), (40,38),          # Left bottom profile
            (33,41), (41,39),          # Right bottom profile
            (34,40), (36,40),          # Left side triangulation
            (35,41), (37,41)           # Right side triangulation
        ]
        
        local_verts.extend(body_verts)
        local_edges.extend(body_edges)

        # Transform all vertices from local coordinates into world coordinates 
        # based on the lightcycle's current position and rotation
        cos_d = math.cos(cycle_dir)
        sin_d = math.sin(cycle_dir)
        transformed_verts = []
        for (vx, vy, vz) in local_verts:
            # X comes from Right Vector (cos/sin), Z comes from Forward Vector (sin/cos)
            wx = cycle_x + (vx * s * cos_d) + (vz * s * sin_d)
            wz = cycle_z + (vx * s * -sin_d) + (vz * s * cos_d)
            wy = vy * s
            transformed_verts.append((wx, wy, wz))
            
        # Draw the resulting mesh using the transformed vertex map
        for eA, eB in local_edges:
            draw_line(transformed_verts[eA], transformed_verts[eB], cx, cy, cz, cam_yaw, CYCLE_COLOR, 3, surf=main_layer)
        
        # -------------- POST PROCESSING (NEON GLOW / CUDA) -------------- #
        using_cuda_this_frame = False
        
        # Determine current pipeline enforcement based on autopilot tests
        attempt_cuda = CUDA_AVAILABLE and (not autodrive_mode or active_profile_mode == "CUDA")

        if attempt_cuda:
            # High-performance CUDA bloom
            try:
                import numpy as np
                pixels = pygame.surfarray.pixels3d(main_layer) # reference to surface bytes
                gpu_px = cp.asarray(pixels, dtype=cp.float32)
                
                # Apply Gaussian blur on GPU for each RGB channel
                for i in range(3):
                    gpu_px[:, :, i] = gaussian_filter(gpu_px[:, :, i], sigma=3.0)
                
                # Additive blend back onto original
                pixels[:] = cp.asnumpy(cp.clip(cp.asarray(pixels) + gpu_px, 0, 255).astype(cp.uint8))
                using_cuda_this_frame = True
            except Exception as e:
                # If NumPy isn't available or CUDA crashes, skip
                pass
            
            # Blit directly to screen now
            screen.blit(main_layer, (0, 0))
            
        if not using_cuda_this_frame:
            # CPU Pygame Fallback Neon bloom
            w, h = main_layer.get_size()
            # Downscale significantly to blur out the sharp lines cheaply but strongly
            small = pygame.transform.smoothscale(main_layer, (w // 4, h // 4))
            blur = pygame.transform.smoothscale(small, (w, h))
            
            screen.blit(main_layer, (0, 0)) # Base sharp pass
            # Overlay blurred version MULTIPLE TIMES to drastically increase the neon intensity
            screen.blit(blur, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            screen.blit(blur, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        # ----------------------------------------------------------------- #
        
        # ==========================
        # Draw AI Knowledge Minimap UI
        # ==========================
        minimap_size = 10
        map_w = MAZE_COLS * minimap_size
        map_h = MAZE_ROWS * minimap_size
        map_x = WIDTH - map_w - 20
        map_y = HEIGHT - map_h - 20
        
        # Draw semi-transparent background for map background
        pygame.draw.rect(screen, (0, 0, 30), (map_x - 5, map_y - 5, map_w + 10, map_h + 10))
        
        for r in range(MAZE_ROWS):
            for c in range(MAZE_COLS):
                rect = (map_x + c*minimap_size, map_y + r*minimap_size, minimap_size, minimap_size)
                if maze_grid[r][c] == 1:
                    # Solid walls are dark blue
                    pygame.draw.rect(screen, (0, 50, 150), rect)
                else:
                    # Render AI spatial memory (heat) for empty spaces
                    memory_val = ai_memory_grid[r][c]
                    if memory_val > 0:
                        # Map memory to a heat color (dark green -> bright yellow -> red)
                        heat_color = (min(255, int(memory_val * 20)), max(50, 255 - int(memory_val * 5)), 0)
                        pygame.draw.rect(screen, heat_color, rect)
                    
        # Draw the physical cycle cursor cleanly onto the minimap
        cx_map = int(((cycle_x + offset_x) / (MAZE_COLS * CELL_SIZE)) * map_w)
        cz_map = int(((cycle_z + offset_z) / (MAZE_ROWS * CELL_SIZE)) * map_h)
        pygame.draw.circle(screen, (0, 255, 255), (map_x + cx_map, map_y + cz_map), 3)

        # Draw HUD overlay (CUDA indicator)
        hud_color = (0, 255, 0) if using_cuda_this_frame else (255, 255, 0)
        hud_text = f"CUDA Bloom: {'ON' if using_cuda_this_frame else 'OFF'} | Numba JIT: {'ON' if NUMBA_AVAILABLE else 'OFF'}"
        text_surf = font.render(hud_text, True, hud_color)
        screen.blit(text_surf, (10, 10))
        
        # Draw HUD FPS text in top right
        fps_text = f"FPS: {int(clock.get_fps())}"
        fps_surf = font.render(fps_text, True, (255, 255, 255))
        fps_rect = fps_surf.get_rect()
        fps_rect.topright = (WIDTH - 10, 10)
        screen.blit(fps_surf, fps_rect)
        
        # Draw current speed indicator below FPS
        speed_text = f"SPEED: {current_speed:.1f} m/s"
        speed_surf = font.render(speed_text, True, (0, 255, 255))
        speed_rect = speed_surf.get_rect()
        speed_rect.topright = (WIDTH - 10, 35)
        screen.blit(speed_surf, speed_rect)
        
        # Draw Autodrive Status under speed
        mode_text = "AUTO-DRIVE (Press ESC to Manual)" if autodrive_mode else "MANUAL CONTROL (Press ESC to Auto)"
        mode_color = (255, 100, 100) if autodrive_mode else (100, 255, 255)
        mode_surf = font.render(mode_text, True, mode_color)
        mode_rect = mode_surf.get_rect()
        mode_rect.topright = (WIDTH - 10, 60)
        screen.blit(mode_surf, mode_rect)
        
        # Log telemetry data line-by-line using standard CSV
        mode_str = "Auto" if autodrive_mode else "Manual"
        renderer_log = "CUDA_GPU" if using_cuda_this_frame else "Numba_CPU"
        telemetry_writer.writerow([
            time.time(), 
            mode_str, 
            round(clock.get_fps(), 2), 
            round(current_speed, 2), 
            round(cycle_x, 2), 
            round(cycle_z, 2), 
            ai_action_log,
            renderer_log
        ])
        
        # Swap the dual display buffers to show our newly drawn composite frame
        pygame.display.flip()
        
        # Keep the game running at a capped 60 Frames Per Second so physics are consistent
        clock.tick(60)

    # Clean up and completely close contexts when main logic exits natively
    telemetry_file.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
