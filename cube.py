import pygame
import numpy as np
import math
from typing import List, Tuple, Any

# Configuration
WIDTH, HEIGHT = 640, 512
FPS = 60
FOCAL_LENGTH = 350
DISTANCE = 400
SIZE = 80

def init_cube(s: float) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Returns a tuple containing:
    - A NumPy array of vertex coordinates.
    - A list of tuples representing edge connections.
    """
    v = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ], dtype=float)
    
    e = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    return v, e

# Optional high-performance backends
HAS_NUMBA = False
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    def njit(func): return func

HAS_TORCH = False
DEVICE = "cpu"
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
except ImportError:
    pass

@njit
def get_rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    """Calculates the 3D rotation matrix using Numba JIT for native performance."""
    c = np.cos(np.array([ax, ay, az]))
    s = np.sin(np.array([ax, ay, az]))
    
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, c[0], -s[0]],
                   [0.0, s[0], c[0]]])
    
    ry = np.array([[c[1], 0.0, s[1]],
                   [0.0, 1.0, 0.0],
                   [-s[1], 0.0, c[1]]])
    
    rz = np.array([[c[2], -s[2], 0.0],
                   [s[2], c[2], 0.0],
                   [0.0, 0.0, 1.0]])
    
    return rz @ ry @ rx

@njit
def fast_project(rotated_vertices: np.ndarray, focal_length: float, distance: float, width: int, height: int) -> np.ndarray:
    """Performs perspective projection using Numba JIT."""
    tx = rotated_vertices[:, 0]
    ty = rotated_vertices[:, 1]
    tz = rotated_vertices[:, 2]
    
    k = focal_length / (tz + distance)
    sx = (width // 2) + tx * k
    sy = (height // 2) - ty * k
    
    # Manually stack for Numba compatibility if needed, though column_stack works in modern numba
    res = np.empty((len(tx), 2))
    res[:, 0] = sx
    res[:, 1] = sy
    return res

def main() -> None:
    """Main animation loop with GPU/Native acceleration."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"3D Cube - Backend: {'CUDA' if DEVICE == 'cuda' else 'Numba' if HAS_NUMBA else 'NumPy'}")
    clock = pygame.time.Clock()

    vertices_np, edges = init_cube(SIZE)
    
    # Prepare PyTorch if available
    vertices_torch = None
    if HAS_TORCH:
        vertices_torch = torch.from_numpy(vertices_np).float().to(DEVICE)

    ax, ay, az = 0.0, 0.0, 0.0
    dax, day, daz = 0.017, 0.021, 0.013

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        rotation_mat_np = get_rotation_matrix(ax, ay, az)
        
        if DEVICE == "cuda":
            # GPU Acceleration via PyTorch
            rot_torch = torch.from_numpy(rotation_mat_np).float().to(DEVICE)
            # rotated = vertices @ rotation_mat.T
            rotated = torch.mm(vertices_torch, rot_torch.t())
            
            tx = rotated[:, 0]
            ty = rotated[:, 1]
            tz = rotated[:, 2]
            
            k = FOCAL_LENGTH / (tz + DISTANCE)
            sx = (WIDTH // 2) + tx * k
            sy = (HEIGHT // 2) - ty * k
            projected_points = torch.stack((sx, sy), dim=1).cpu().numpy()
        else:
            # CPU Acceleration via Numba or NumPy
            rotated = vertices_np @ rotation_mat_np.T
            if HAS_NUMBA:
                projected_points = fast_project(rotated, FOCAL_LENGTH, DISTANCE, WIDTH, HEIGHT)
            else:
                # Fallback to pure NumPy vectorized (already fast, but not JIT'd)
                tx, ty, tz = rotated.T
                k = FOCAL_LENGTH / (tz + DISTANCE)
                sx = (WIDTH // 2) + tx * k
                sy = (HEIGHT // 2) - ty * k
                projected_points = np.column_stack((sx, sy))

        # Draw the wireframe
        for edge in edges:
            p1, p2 = projected_points[edge[0]], projected_points[edge[1]]
            pygame.draw.line(screen, (255, 255, 255), p1, p2, 1)

        pygame.display.flip()
        ax += dax; ay += day; az += daz
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
