"""Predict each shape, run the script, then try the exercises at the bottom."""

import torch


def main():
    # Each row is a day, and each column is a station. The decimals make
    # this a floating-point tensor, which lets us compute averages.
    readings = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    print("Readings", tuple(readings.shape), readings)  # (2, 3)

    # An integer selects one row. That axis disappears from the result.
    first_day = readings[0]
    print("First day", tuple(first_day.shape), first_day)  # (3,)

    # The colon selects every day. The integer selects the first station.
    first_station = readings[:, 0]
    print("First station", tuple(first_station.shape), first_station)  # (2,)

    # A slice selects a range and keeps the axis, even for a single station.
    station_column = readings[:, :1]
    print("Station column", tuple(station_column.shape), station_column)  # (2, 1)

    # dim=0 averages down the rows, leaving one result per station.
    station_means = readings.mean(dim=0)
    print("Mean per station", tuple(station_means.shape), station_means)  # (3,)

    # dim=1 averages across columns, leaving one result per day.
    day_means = readings.mean(dim=1)
    print("Mean per day", tuple(day_means.shape), day_means)  # (2,)

    # With no dim, mean averages all six readings into a scalar.
    overall_mean = readings.mean()
    print("Overall mean", tuple(overall_mean.shape), overall_mean)  # ()

    # Insert a length-one axis at position 1 to make a column.
    day_mean_column = day_means.unsqueeze(1)
    print("Day mean column", tuple(day_mean_column.shape), day_mean_column)  # (2, 1)

    # Broadcasting uses each day's single mean for all three stations.
    centered = readings - day_mean_column
    print("Centered readings", tuple(centered.shape), centered)  # (2, 3)

    # Try it yourself. Predict both the shape and values before running.
    # 1. Select the last station for both days. Then keep its column axis.
    #    Hint: compare readings[:, -1] with readings[:, -1:].
    # 2. Add a third day. Which printed shapes change? Which stay the same?
    # 3. Subtract station_means from readings. Which way does broadcasting
    #    repeat the means this time? Average over days to check your result.


if __name__ == "__main__":
    main()
