def fit_nurbs_surface(
    self,
    points: np.ndarray,
    n_grid: int = 50,
    smoothing: float = None,
):
    """
    用B-spline拟合点云生成NURBS曲面

    方法: PCA找主平面 → 在主平面上参数化 → bisplrep拟合 → 采样生成网格

    Args:
        points: (M, 3) 待拟合的3D点
        n_grid: 生成网格的分辨率 (n_grid x n_grid)
        smoothing: 平滑参数, None则自动设置

    Returns:
        grid_points: (n_grid*n_grid, 3) 拟合曲面上的采样点
        triangles: (K, 3) 三角面片索引
        info: dict 拟合信息
    """
    if len(points) < 16:
        print(f"  [NURBS] 点数太少({len(points)})，无法拟合曲面")
        return None, None, None

    # 1. PCA参数化
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    cov = np.cov(pts_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 按特征值降序排列, 前两个主方向为曲面面内方向, 第三个为法方向
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 投影到主方向坐标系: u=PC1, v=PC2, w=PC3(法方向)
    local_coords = pts_centered @ eigenvectors  # (M, 3)
    u_coords = local_coords[:, 0]
    v_coords = local_coords[:, 1]
    w_coords = local_coords[:, 2]

    # 2. B-spline拟合 w = f(u, v)
    if smoothing is None:
        smoothing = len(points) * 0.1  # 适度平滑

    try:
        tck = bisplrep(u_coords, v_coords, w_coords, s=smoothing, kx=3, ky=3)
    except Exception as e:
        print(f"  [NURBS] bisplrep拟合失败: {e}")
        # 尝试更大的平滑参数
        try:
            tck = bisplrep(u_coords, v_coords, w_coords,
                            s=len(points) * 1.0, kx=3, ky=3)
            print(f"  [NURBS] 使用更大平滑参数重试成功")
        except Exception as e2:
            print(f"  [NURBS] 重试也失败: {e2}")
            return None, None, None

    # 3. 在均匀网格上求值
    u_margin = (u_coords.max() - u_coords.min()) * 0.02
    v_margin = (v_coords.max() - v_coords.min()) * 0.02
    u_grid = np.linspace(u_coords.min() + u_margin, u_coords.max() - u_margin, n_grid)
    v_grid = np.linspace(v_coords.min() + v_margin, v_coords.max() - v_margin, n_grid)

    w_grid = bisplev(u_grid, v_grid, tck)  # (n_grid, n_grid)

    # 4. 转换回原始坐标系
    grid_points = []
    for i, u in enumerate(u_grid):
        for j, v in enumerate(v_grid):
            local_pt = np.array([u, v, w_grid[i, j]])
            world_pt = local_pt @ eigenvectors.T + centroid
            grid_points.append(world_pt)

    grid_points = np.array(grid_points)  # (n_grid*n_grid, 3)

    # 5. 生成三角网格索引
    triangles = []
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            idx = i * n_grid + j
            # 两个三角形组成一个四边形
            triangles.append([idx, idx + 1, idx + n_grid])
            triangles.append([idx + 1, idx + n_grid + 1, idx + n_grid])

    triangles = np.array(triangles)

    info = {
        'n_input_points': len(points),
        'n_grid': n_grid,
        'eigenvalues': eigenvalues,
        'flatness_ratio': eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0,
    }

    print(f"  [NURBS] 拟合完成: {n_grid}x{n_grid}网格, "
            f"平坦度={info['flatness_ratio']:.4f}")

    return grid_points, triangles, info

def create_nurbs_mesh_o3d(
    self,
    grid_points: np.ndarray,
    triangles: np.ndarray,
    color: Tuple[float, float, float] = (0.0, 0.8, 0.4),
    opacity: float = 0.7,
):
    """
    从拟合结果创建Open3D TriangleMesh

    Args:
        grid_points: (V, 3) 顶点
        triangles: (F, 3) 三角面片
        color: 曲面颜色
        opacity: 透明度 (Open3D基本可视化不直接支持, 用于未来扩展)

    Returns:
        mesh: Open3D TriangleMesh
    """
    if not self.use_open3d:
        return None

    mesh = self.o3d.geometry.TriangleMesh()
    mesh.vertices = self.o3d.utility.Vector3dVector(grid_points.astype(np.float64))
    mesh.triangles = self.o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh