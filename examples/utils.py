"""Utility functions for examples."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import swe_simulator.utils as sim_utils


def animate_surface_from_output(
    output_path: str,
    frames: list[int] | None = None,
    wave_treshold: float = 1e-3,
    interval: int = 50,
    elev: float = 30.0,
    azim: float = -120.0,
    save: bool = False,
) -> None:
    """Create a 3D surface animation of wave solutions from Clawpack output.

    Parameters
    ----------
    output_path : str
        Path to the Clawpack output directory.
    frames : list[int] | None
        List of frame indices to animate, or None for all frames.
    wave_treshold : float
        Threshold below which water height is masked (default: 1e-3).
    interval : int
        Interval between frames in milliseconds (default: 50).
    elev : float
        Camera elevation angle in degrees (default: 30.0).
    azim : float
        Camera azimuth angle in degrees (default: -120.0).
    save : bool
        If True, save animation as MP4; otherwise display interactively.
    """
    import matplotlib.animation as animation

    result = sim_utils.io.read_solutions(output_path, frames_list=frames)
    bathymetry = result["bathymetry"]
    X, Y = result["meshgrid"]
    solutions = result["solutions"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cax = fig.add_axes((0.88, 0.16, 0.03, 0.68))
    fig.subplots_adjust(right=0.86)

    x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
    y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
    z_min = float(np.nanmin(bathymetry))
    z_max = float(np.nanmax(bathymetry + solutions[:, 0, :, :]))

    colorbar: Any | None = None

    def update(frame_idx: int) -> tuple[Any]:
        nonlocal colorbar
        ax.clear()
        cax.clear()

        sol = solutions[frame_idx]
        h = sol[0, :, :]
        free_surface = bathymetry + h
        free_surface[h < wave_treshold] = np.nan

        surface = ax.plot_surface(
            X,
            Y,
            free_surface,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
            alpha=0.95,
            vmin=float(np.nanmin(free_surface)),
            vmax=float(np.nanmax(free_surface)),
        )

        ax.plot_surface(
            X,
            Y,
            bathymetry,
            cmap="terrain",
            linewidth=0,
            antialiased=False,
            alpha=0.8,
        )
        colorbar = fig.colorbar(surface, cax=cax, label="Surface Elevation (m)")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min - 0.1, z_max + 0.1)
        ax.set_zlabel("Surface Elevation (m)")
        ax.set_title(f"Wave surface at frame {frame_idx}")
        ax.view_init(elev=elev, azim=azim)
        return (surface,)

    ani = animation.FuncAnimation(
        fig, update, frames=solutions.shape[0], interval=interval
    )

    if save:
        ani.save(f"{output_path}/wave_surface_animation.mp4", writer="ffmpeg", dpi=200)
    else:
        plt.show()
