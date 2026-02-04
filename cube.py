import pyray as pr
import numpy as np
import math
from typing import List, Tuple, Any

# -- CONFIGURATION & RENDERING PARAMETERS --
WIDTH, HEIGHT = 640, 512
FPS = 60
FOCAL_LENGTH = 350 # Controls the perspective 'zoom'
DISTANCE = 400     # Distance from camera to cube center
SIZE = 80          # Cube half-side length

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

# ---------------------------------------------------------
# BACKEND DETECTION & ACCELERATION
# ---------------------------------------------------------
# We check for Numba (CPU JIT) and PyTorch (GPU CUDA) to select the fastest path.

HAS_NUMBA = False
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    # No-op decorator if Numba is missing
    def njit(func): return func

HAS_TORCH = False
DEVICE = "cpu"
try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"Check: CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Check: CUDA is NOT available. Falling back to CPU.")
except ImportError:
    print("Check: PyTorch not installed.")
    pass

# ---------------------------------------------------------
# CPU / NUMBA FUNCTIONS
# ---------------------------------------------------------
@njit
def get_rotation_matrix_cpu(ax: float, ay: float, az: float) -> np.ndarray:
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
def fast_project_cpu(rotated_vertices: np.ndarray, focal_length: float, distance: float, width: int, height: int) -> np.ndarray:
    tx = rotated_vertices[:, 0]
    ty = rotated_vertices[:, 1]
    tz = rotated_vertices[:, 2]
    
    k = focal_length / (tz + distance)
    sx = (width // 2) + tx * k
    sy = (height // 2) - ty * k
    
    res = np.empty((len(tx), 2))
    res[:, 0] = sx
    res[:, 1] = sy
    return res

@njit
def get_face_shading_cpu(rotated_vertices: np.ndarray, faces: np.ndarray, light_dir: np.ndarray) -> np.ndarray:
    shading = np.zeros(len(faces))
    for i in range(len(faces)):
        v0 = rotated_vertices[faces[i, 0]]
        v1 = rotated_vertices[faces[i, 1]]
        v2 = rotated_vertices[faces[i, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normal = np.array([
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ])
        
        norm = np.sqrt(np.sum(normal**2))
        if norm > 0:
            normal = normal / norm
            
        dot = np.sum(normal * light_dir)
        shading[i] = max(0.1, dot)
    return shading

# ---------------------------------------------------------
# GPU / TORCH FUNCTIONS
# ---------------------------------------------------------
def get_rotation_matrix_torch(ax, ay, az, device):
    """
    Constructs a 3D rotation matrix using PyTorch tensors.
    Combines rotations around X, Y, and Z axes.
    """
    ax_t = torch.tensor(ax, device=device)
    ay_t = torch.tensor(ay, device=device)
    az_t = torch.tensor(az, device=device)
    
    c = torch.cos(torch.stack([ax_t, ay_t, az_t]))
    s = torch.sin(torch.stack([ax_t, ay_t, az_t]))

    rx = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, c[0], -s[0]],
                       [0.0, s[0], c[0]]], device=device)
    
    ry = torch.tensor([[c[1], 0.0, s[1]],
                       [0.0, 1.0, 0.0],
                       [-s[1], 0.0, c[1]]], device=device)
    
    rz = torch.tensor([[c[2], -s[2], 0.0],
                       [s[2], c[2], 0.0],
                       [0.0, 0.0, 1.0]], device=device)

    # Combine rotations via matrix multiplication
    return torch.matmul(rz, torch.matmul(ry, rx))

def get_face_shading_torch(rotated_vertices, faces, light_dir):
    """
    Calculates face shading (Lambertian reflectance) using GPU-accelerated vector operations.
    Computes normals via cross product and dots them with the light source.
    """
    # Gather vertices for all faces at once
    # faces shape: [num_faces, 4]
    # v0, v1, v2 shape: [num_faces, 3]
    v0 = rotated_vertices[faces[:, 0]]
    v1 = rotated_vertices[faces[:, 1]]
    v2 = rotated_vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Cross product
    normal = torch.cross(edge1, edge2, dim=1)
    
    # Normalize
    norm = torch.norm(normal, dim=1, keepdim=True)
    # Avoid division by zero
    normal = torch.where(norm > 0, normal / norm, normal)
    
    # Dot product with light
    # normal: [num_faces, 3], light_dir: [3]
    dot = torch.matmul(normal, light_dir)
    return torch.clamp(dot, min=0.1)

def fast_project_torch(rotated_vertices, focal_length, distance, width, height):
    tx = rotated_vertices[:, 0]
    ty = rotated_vertices[:, 1]
    tz = rotated_vertices[:, 2]
    
    k = focal_length / (tz + distance)
    sx = (width // 2) + tx * k
    sy = (height // 2) - ty * k
    
    return torch.stack([sx, sy], dim=1)


def main() -> None:
    global DEVICE
    # Initialize Raylib
    pr.init_window(WIDTH, HEIGHT, f"3D Cube - Backend: {'CUDA' if DEVICE == 'cuda' else 'Numba' if HAS_NUMBA else 'NumPy'}")
    pr.set_target_fps(FPS)

    vertices_np, faces_list, edges = init_cube(SIZE)
    faces_np = np.array(faces_list)
    light_dir = np.array([0.5, 0.5, -1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # GPU Tensors Initialization
    vertices_torch = None
    faces_torch = None
    light_dir_torch = None
    
    if HAS_TORCH and DEVICE == 'cuda':
        try:
            vertices_torch = torch.from_numpy(vertices_np).float().to(DEVICE)
            faces_torch = torch.tensor(faces_list, dtype=torch.long, device=DEVICE)
            light_dir_torch = torch.from_numpy(light_dir).float().to(DEVICE)
            print("PyTorch Tensors initialized on GPU.")
        except Exception as e:
            print(f"Failed to initialize CUDA tensors: {e}")
            # global DEVICE # Not needed in this scope as we are reading it or it's module level
            DEVICE = 'cpu' # Fallback

    ax, ay, az = 1.0, 1.0, 0.0
    dax, day, daz = 0.017, 0.021, 0.013

    bg_color = pr.Color(10, 10, 20, 255)

    while not pr.window_should_close():
        # Input
        if pr.is_key_pressed(pr.KeyboardKey.KEY_UP): dax += 0.005
        if pr.is_key_pressed(pr.KeyboardKey.KEY_DOWN): dax -= 0.005
        if pr.is_key_pressed(pr.KeyboardKey.KEY_LEFT): day -= 0.005
        if pr.is_key_pressed(pr.KeyboardKey.KEY_RIGHT): day += 0.005
        if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE): dax, day, daz = 0, 0, 0

        # DATA PROCESSING BRANCH
        if DEVICE == 'cuda':
            # --- GPU PATH ---
            rot_mat = get_rotation_matrix_torch(ax, ay, az, DEVICE)
            rotated_t = torch.matmul(vertices_torch, rot_mat.T)
            
            shading_t = get_face_shading_torch(rotated_t, faces_torch, light_dir_torch)
            projected_t = fast_project_torch(rotated_t, FOCAL_LENGTH, DISTANCE, WIDTH, HEIGHT)
            
            # Sort faces (Z-sort) on CPU for now as it's easier to iterate
            # Technically we could sort on GPU but transferring a small list of indices is fast
            
            # Transfer required data to CPU for drawing
            # We need: projected points, shading values, and efficient face drawing
            projected_points = projected_t.cpu().numpy()
            shading = shading_t.cpu().numpy()
            
            # For sorting we need Z separation
            # Calculate mean Z per face
            # Gather Z values: [num_faces, 4]
            z_vals = rotated_t[faces_torch, 2] 
            z_avg = torch.mean(z_vals, dim=1)
            z_avg_cpu = z_avg.cpu().numpy()
            
            # Create the list for sorting
            face_z = []
            for i in range(len(faces_list)):
                face_z.append((z_avg_cpu[i], i))

        else:
            # --- CPU/NUMBA PATH ---
            rotation_mat_np = get_rotation_matrix_cpu(ax, ay, az)
            rotated = vertices_np @ rotation_mat_np.T
            
            if HAS_NUMBA:
                shading = get_face_shading_cpu(rotated, faces_np, light_dir)
                projected_points = fast_project_cpu(rotated, FOCAL_LENGTH, DISTANCE, WIDTH, HEIGHT)
            else:
                # Fallback Pure NumPy
                shading = np.ones(len(faces_np)) * 0.5
                tx, ty, tz = rotated.T
                k = FOCAL_LENGTH / (tz + DISTANCE)
                sx = (WIDTH // 2) + tx * k
                sy = (HEIGHT // 2) - ty * k
                projected_points = np.column_stack((sx, sy))
                
            # Face Sorting
            face_z = []
            for i, face in enumerate(faces_list):
                z_avg = np.mean(rotated[face, 2])
                face_z.append((z_avg, i))

        # Sort Faces
        face_z.sort(key=lambda x: x[0], reverse=True)

        # RENDER
        pr.begin_drawing()
        pr.clear_background(bg_color)

        for _, i in face_z:
            face = faces_list[i]
            points = [pr.Vector2(projected_points[v][0], projected_points[v][1]) for v in face]
            s = shading[i]
            
            r, g, b = int(100 * s), int(150 * s), int(255 * s)
            face_color = pr.Color(r, g, b, 255)
            edge_color = pr.Color(min(255, r + 50), min(255, g + 50), min(255, b + 50), 255)

            pr.draw_triangle_fan(points, len(points), face_color)
            for j in range(len(points)):
                pr.draw_line_v(points[j], points[(j+1)%len(points)], edge_color)

        pr.draw_fps(10, 10)
        pr.end_drawing()

        ax += dax; ay += day; az += daz

    pr.close_window()

if __name__ == "__main__":
    main()
