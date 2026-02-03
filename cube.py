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

def init_cube(s: float) -> Tuple[np.ndarray, List[List[int]], List[Tuple[int, int]]]:
    """
    Returns a tuple containing:
    - A NumPy array of vertex coordinates.
    - A list of lists representing face (vertex indices).
    - A list of tuples representing edge connections.
    """
    v = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]
    ], dtype=float)
    
    # Faces defined by vertex indices (counter-clockwise)
    f = [
        [0, 1, 2, 3], # Back
        [4, 5, 6, 7], # Front
        [0, 4, 7, 3], # Left
        [1, 5, 6, 2], # Right
        [0, 1, 5, 4], # Bottom
        [3, 2, 6, 7]  # Top
    ]
    
    e = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    return v, f, e

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

@njit
def get_face_shading(rotated_vertices: np.ndarray, faces: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
    """Calculates shading for each face based on light direction."""
    shading = np.zeros(len(faces))
    for i in range(len(faces)):
        # Calculate normal using cross product of two edges
        v0 = rotated_vertices[faces[i, 0]]
        v1 = rotated_vertices[faces[i, 1]]
        v2 = rotated_vertices[faces[i, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Manual cross product for Numba
        normal = np.array([
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ])
        
        # Normalize
        norm = np.sqrt(np.sum(normal**2))
        if norm > 0:
            normal = normal / norm
            
        # Dot product with light
        dot = np.sum(normal * light_dir)
        shading[i] = max(0.1, dot) # Ambient floor
    return shading

def main() -> None:
    """Main animation loop with GPU/Native acceleration."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"3D Cube - Backend: {'CUDA' if DEVICE == 'cuda' else 'Numba' if HAS_NUMBA else 'NumPy'}")
    clock = pygame.time.Clock()

    vertices_np, faces_list, edges = init_cube(SIZE)
    faces_np = np.array(faces_list)
    light_dir = np.array([0.5, 0.5, -1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Prepare PyTorch if available
    vertices_torch = None
    if HAS_TORCH:
        vertices_torch = torch.from_numpy(vertices_np).float().to(DEVICE)

    ax, ay, az = 1.0, 1.0, 0.0 # Better start angle
    dax, day, daz = 0.017, 0.021, 0.013

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Interactive Controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: dax += 0.005
                if event.key == pygame.K_DOWN: dax -= 0.005
                if event.key == pygame.K_LEFT: day -= 0.005
                if event.key == pygame.K_RIGHT: day += 0.005
                if event.key == pygame.K_SPACE: dax, day, daz = 0, 0, 0 # Stop rotation

        screen.fill((10, 10, 20)) # Dark blue background
        rotation_mat_np = get_rotation_matrix(ax, ay, az)
        
        # CPU Rotation
        rotated = vertices_np @ rotation_mat_np.T
        
        # Shading
        if HAS_NUMBA:
            shading = get_face_shading(rotated, faces_np, light_dir)
            projected_points = fast_project(rotated, FOCAL_LENGTH, DISTANCE, WIDTH, HEIGHT)
        else:
            # Fallback (simplified)
            shading = np.ones(len(faces_np)) * 0.5
            tx, ty, tz = rotated.T
            k = FOCAL_LENGTH / (tz + DISTANCE)
            sx = (WIDTH // 2) + tx * k
            sy = (HEIGHT // 2) - ty * k
            projected_points = np.column_stack((sx, sy))

        # Face Sorting (Painters Algorithm for solid fill)
        face_z = []
        for i, face in enumerate(faces_list):
            z_avg = np.mean(rotated[face, 2])
            face_z.append((z_avg, i))
        
        # Sort faces: furthest first
        face_z.sort(key=lambda x: x[0], reverse=True)

        # Draw the solid faces
        for _, i in face_z:
            face = faces_list[i]
            points = [projected_points[v] for v in face]
            s = shading[i]
            color = (int(100 * s), int(150 * s), int(255 * s)) # Gradient blue
            pygame.draw.polygon(screen, color, points)
            # Optional: Draw edges with slightly lighter color
            pygame.draw.polygon(screen, (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50)), points, 1)

        pygame.display.flip()
        ax += dax; ay += day; az += daz
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
