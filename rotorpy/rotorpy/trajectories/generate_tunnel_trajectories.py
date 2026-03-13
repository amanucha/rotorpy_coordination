import numpy as np
from rotorpy.trajectories.mixed_traj import PiecewiseTrajectory

def generate_tunnel_trajectories():
    trajectories = []
    
    TUNNEL_START_X = 25.0
    TUNNEL_END_X = 40.0
    TUNNEL_Y = 0.0
    TUNNEL_Z = 1.0
    
    FINAL_TARGET_X = 65.0
    
    # Jittered starting positions to break alignment on X=0
    x_start_offsets = [0.0, 10.0, -12.0, 6.0]
    # Non-symmetric offsets to avoid pair-symmetry look
    y_start_offsets = [-5.5, -4.2, 4.8, 7] 
    z_start_offsets = [0.2, 0, 0.2, 0.1]
    
    # Non-symmetric final lanes
    y_final_offsets = [-3.6, -1.3, 0.4, 2.7] 
    z_final_offsets = [1.0, 1.0, 1.0, 1.0] 

    TOTAL_FLIGHT_DURATION = 60.0
    APPROACH_DURATION = 25.0
    TUNNEL_DURATION = 15.0
    DIVERGE_DURATION = 10.0
    FINAL_DURATION = 10.0

    # Sanity check
    assert np.isclose(APPROACH_DURATION + TUNNEL_DURATION + DIVERGE_DURATION + FINAL_DURATION, TOTAL_FLIGHT_DURATION)
    
    def calculate_cubic_coeffs(P0, V0, Pf, Vf, T):
        a0 = P0
        a1 = V0
        T2, T3 = T**2, T**3
        a2 = 3.0 * (Pf - P0) / T2 - (2.0 * V0 + Vf) / T
        a3 = 2.0 * (P0 - Pf) / T3 + (V0 + Vf) / T2
        return a0, a1, a2, a3
        
    # Set seeds for consistent trajectory shapes
    np.random.seed(42)
    
    for i in range(4):
        segments = []
        
        # Start state
        p_start = np.array([x_start_offsets[i], y_start_offsets[i], z_start_offsets[i]])
        
        # Velocities are now segment-specific to ensure fixed durations
        v_forward = np.array([(TUNNEL_END_X - TUNNEL_START_X) / TUNNEL_DURATION, 0.0, 0.0])

        # Tunnel start state
        p_tunnel_start = np.array([TUNNEL_START_X, TUNNEL_Y, TUNNEL_Z])
        
        # Segment 1: Approach to tunnel (simplified 3 sub-segments, no extra oscillations)
        # Use fixed duration for approach
        approach_dist = TUNNEL_START_X - p_start[0]
        drone_approach_duration = APPROACH_DURATION
        drone_vx_approach = approach_dist / drone_approach_duration

        wp1_x = p_start[0] + approach_dist * 0.33
        wp1_y = np.interp(wp1_x, [p_start[0], TUNNEL_START_X], [p_start[1], TUNNEL_Y]) + (np.random.random()-0.5)*1.0
        wp1_z = np.interp(wp1_x, [p_start[0], TUNNEL_START_X], [p_start[2], TUNNEL_Z]) + (np.random.random()-0.5)*1.5
        p_wp1 = np.array([wp1_x, wp1_y, max(0.5, wp1_z)])
        
        wp2_x = p_start[0] + approach_dist * 0.66
        wp2_y = np.interp(wp2_x, [p_start[0], TUNNEL_START_X], [p_start[1], TUNNEL_Y]) + (np.random.random()-0.5)*1.0
        wp2_z = np.interp(wp2_x, [p_start[0], TUNNEL_START_X], [p_start[2], TUNNEL_Z]) + (np.random.random()-0.5)*1.5
        p_wp2 = np.array([wp2_x, wp2_y, max(0.5, wp2_z)])
        
        # Velocities for approach waypoints
        # Use drone-specific approach duration for velocity estimates
        v_wp1 = np.array([drone_vx_approach, (p_wp2[1]-p_start[1])/(drone_approach_duration*0.66), (p_wp2[2]-p_start[2])/(drone_approach_duration*0.66)])
        v_wp2 = np.array([drone_vx_approach, (p_tunnel_start[1]-p_wp1[1])/(drone_approach_duration*0.66), (p_tunnel_start[2]-p_wp1[2])/(drone_approach_duration*0.66)])
        
        v_start = np.array([drone_vx_approach, 0.0, 0.0])

        t_ratios = np.random.dirichlet(np.ones(3)*2, size=1)[0]
        
        # Approach Segment A
        a0, a1, a2, a3 = calculate_cubic_coeffs(p_start, v_start, p_wp1, v_wp1, drone_approach_duration*t_ratios[0])
        segments.append({"duration": drone_approach_duration*t_ratios[0], "type": "cubic", "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}})
        
        # Approach Segment B
        a0, a1, a2, a3 = calculate_cubic_coeffs(p_wp1, v_wp1, p_wp2, v_wp2, drone_approach_duration*t_ratios[1])
        segments.append({"duration": drone_approach_duration*t_ratios[1], "type": "cubic", "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}})
        
        # Approach Segment C
        a0, a1, a2, a3 = calculate_cubic_coeffs(p_wp2, v_wp2, p_tunnel_start, v_forward, drone_approach_duration*t_ratios[2])
        segments.append({"duration": drone_approach_duration*t_ratios[2], "type": "cubic", "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}})
        
        # Segment 2: The Tunnel (linear, shared by all)
        segments.append({
            "duration": TUNNEL_DURATION,
            "type": "line",
            "params": {"p0": p_tunnel_start, "v": v_forward}
        })
        
        # Segment 3: Diverge from tunnel to parallel tracks
        p_tunnel_end = np.array([TUNNEL_END_X, TUNNEL_Y, TUNNEL_Z])
        
        PARALLEL_START_X = TUNNEL_END_X + (FINAL_TARGET_X - TUNNEL_END_X) * 0.4
        diverge_phase_duration = DIVERGE_DURATION
        parallel_phase_duration = FINAL_DURATION

        drone_vx_diverge = (PARALLEL_START_X - TUNNEL_END_X) / diverge_phase_duration
        drone_vx_final = (FINAL_TARGET_X - PARALLEL_START_X) / parallel_phase_duration
        v_final = np.array([drone_vx_final, 0.0, 0.0])
        
        p_parallel_start = np.array([PARALLEL_START_X, y_final_offsets[i], z_final_offsets[i]])
        
        # Diverge Waypoint
        wp3_x = TUNNEL_END_X + (PARALLEL_START_X - TUNNEL_END_X) * 0.5
        wp3_y = np.interp(wp3_x, [TUNNEL_END_X, PARALLEL_START_X], [TUNNEL_Y, y_final_offsets[i]]) + (np.random.random()-0.5)*1.5
        wp3_z = np.interp(wp3_x, [TUNNEL_END_X, PARALLEL_START_X], [TUNNEL_Z, z_final_offsets[i]]) + (np.random.random()-0.5)*1.5
        p_wp3 = np.array([wp3_x, wp3_y, max(0.5, wp3_z)])
        
        # Velocities for diverge waypoint
        v_wp3 = np.array([drone_vx_diverge, (p_parallel_start[1]-p_tunnel_end[1])/(diverge_phase_duration), 0.0]) # force level

        t_ratios2 = np.random.dirichlet(np.ones(2)*2, size=1)[0]
        
        # Diverge Segment A
        a0, a1, a2, a3 = calculate_cubic_coeffs(p_tunnel_end, v_forward, p_wp3, v_wp3, diverge_phase_duration*t_ratios2[0])
        segments.append({"duration": diverge_phase_duration*t_ratios2[0], "type": "cubic", "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}})
        
        # Diverge Segment B
        a0, a1, a2, a3 = calculate_cubic_coeffs(p_wp3, v_wp3, p_parallel_start, v_final, diverge_phase_duration*t_ratios2[1])
        segments.append({"duration": diverge_phase_duration*t_ratios2[1], "type": "cubic", "params": {"a0": a0, "a1": a1, "a2": a2, "a3": a3}})
        
        # Segment 4: Parallel flight to destination
        segments.append({
            "duration": parallel_phase_duration,
            "type": "line",
            "params": {"p0": p_parallel_start, "v": v_final}
        })
            
        trajectories.append(PiecewiseTrajectory(segments))
        
    return trajectories

def draw_building(ax, x, y, dx, dy, min_z, max_z, color='gray', theta=np.pi/2):
    # Center of rotation (center of the bottom face)
    cx, cy = x + dx/2.0, y + dy/2.0
    c, s = np.cos(theta), np.sin(theta)
    
    # Local coordinate corners relative to center
    l_corners = [
        (-dx/2.0, -dy/2.0),
        ( dx/2.0, -dy/2.0),
        ( dx/2.0,  dy/2.0),
        (-dx/2.0,  dy/2.0)
    ]
    
    # Rotate and translate back to global
    g_corners = []
    for lx, ly in l_corners:
        gx = cx + lx * c - ly * s
        gy = cy + lx * s + ly * c
        g_corners.append((gx, gy))
        
    # Create the 2x2 grid for top/bottom surfaces
    X = np.array([[g_corners[0][0], g_corners[1][0]], 
                  [g_corners[3][0], g_corners[2][0]]])
    Y = np.array([[g_corners[0][1], g_corners[1][1]], 
                  [g_corners[3][1], g_corners[2][1]]])
    
    Z_bottom = np.array([[min_z, min_z], [min_z, min_z]])
    Z_top = np.array([[max_z, max_z], [max_z, max_z]])
    
    # Bottom + Top faces
    ax.plot_surface(X, Y, Z_bottom, color=color, alpha=0.5)
    ax.plot_surface(X, Y, Z_top, color=color, alpha=0.5)

    # Side faces
    xs = [p[0] for p in g_corners] + [g_corners[0][0]]
    ys = [p[1] for p in g_corners] + [g_corners[0][1]]
    
    for i in range(4):
        x_face = np.array([[xs[i], xs[i+1]], [xs[i], xs[i+1]]])
        y_face = np.array([[ys[i], ys[i+1]], [ys[i], ys[i+1]]])
        z_face = np.array([[min_z, min_z], [max_z, max_z]])
        ax.plot_surface(x_face, y_face, z_face, color=color, alpha=0.5)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    trajectories = generate_tunnel_trajectories()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-calculate trajectory points for distance checking
    # Evaluate each trajectory only up to its valid duration to avoid lines snapping back to origin
    all_traj_pts = []
    for i, traj in enumerate(trajectories):
        t_final = sum(s['duration'] for s in traj.segments)
        print(f"Traj {i+1} duration: {t_final:.2f}s")
        t_eval = np.linspace(0, t_final, 500)
        pts = np.array([np.squeeze(traj.update(t)["x"][:3]) for t in t_eval])
        all_traj_pts.append(pts)
        
    for i, pts in enumerate(all_traj_pts):
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=f'Traj {i+1}')
    
    # Plot obstacle buildings bordering the paths
    buildings = []
    
    np.random.seed(42) # Consistent buildings
    # Generate random buildings strictly localized around the tunnel gap
    attempts = 0
    # Increased building count for the longer path
    while len(buildings) < 120 and attempts < 3000:
        attempts += 1
        # Spread buildings across the active flight area
        bx = np.random.uniform(-10.0, 60.0)
        
        # Random Y bounds across the whole width
        by = np.random.uniform(-10.0, 10.0)
        
        # Keep things out of the absolute dead-center (0.0) tunnel
        if 24.5 < bx < 40.5 and -2.5 < by < 2.5:
            by = np.sign(by) * 3.5 # push it out of the tunnel track
            
        b_width = np.random.uniform(0.5, 2.0)
        b_depth = np.random.uniform(0.5, 2.0)
        b_height = np.random.uniform(2.0, 10.0)
        
        # Check if building is close to ANY trajectory point
        b_center_np = np.array([bx + b_width/2, by + b_depth/2])
        is_close = False
        
        for pts in all_traj_pts:
            xy_pts = pts[:, :2]
            distances = np.linalg.norm(xy_pts - b_center_np, axis=1)
            if np.min(distances) < 3.0:
                is_close = True
                break
                
        if is_close:
            buildings.append((bx, by, b_width, b_depth, 0.0, b_height))
        
    # for b in buildings:
    #     draw_building(ax, *b)

    # Adjust axes limits so we view the paths from afar with generous proportions
    ax.set_xlim(-15, 70)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-0.5, 3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Extended Tunnel Trajectories with Small Obstacles')
    plt.show()
