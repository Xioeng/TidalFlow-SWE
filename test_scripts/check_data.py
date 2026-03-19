import argparse

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot monitoring stations from CSV file"
    )
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    args = parser.parse_args()

    # Read CSV file
    df = pd.read_csv(args.csv_path)

    # Get unique locations
    locations = df[
        ["latitude_decimal_degrees", "longitude_decimal_degrees"]
    ].drop_duplicates()

    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Plot stations
    ax.scatter(
        locations["longitude_decimal_degrees"],
        locations["latitude_decimal_degrees"],
        transform=ccrs.PlateCarree(),
        s=50,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )

    ax.set_title("Monitoring Stations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
