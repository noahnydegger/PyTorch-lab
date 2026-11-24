"""Change the synthetic test flows and compare next-day discharge forecasts."""

import torch


def main():
    # Try holding the final three flows steady, or adding a sharp rise there.
    # All entries are daily mean discharge in m³/s.
    q = torch.tensor([2., 3., 7., 5., 3., 2., 4., 9., 6., 4., 3., 2., 4., 8., 8., 7., 6.])
    today, tomorrow = q[:-1], q[1:]
    design = torch.stack((today, torch.ones_like(today)), dim=1)  # (16, 2)
    slope, intercept = torch.linalg.lstsq(design[:10], tomorrow[:10]).solution

    for period, rows in (("Validation", slice(10, 13)), ("Test", slice(13, None))):
        actual = tomorrow[rows]
        persistence = today[rows]
        linear = slope * today[rows] + intercept
        print(f"{period} MAE (m³/s)")
        print(f"  Persistence {(persistence - actual).abs().mean():.2f}")
        print(f"  Linear      {(linear - actual).abs().mean():.2f}")


if __name__ == "__main__":
    main()
