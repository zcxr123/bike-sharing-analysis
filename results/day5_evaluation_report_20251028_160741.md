# Day 5 策略评估报告

生成时间: 2025-10-28 16:07:41

## 策略对比

| scenario      | policy                 |   service_rate |   net_profit |   total_cost |
|:--------------|:-----------------------|---------------:|-------------:|-------------:|
| default       | Zero-Action            |       0.942346 |       124635 |         0    |
| default       | Proportional-Optimized |       0.999904 |       130903 |      1152.22 |
| default       | Min-Cost-Flow          |       0.951673 |       125675 |         0    |
| sunny_weekday | Zero-Action            |       0.942346 |       124635 |         0    |
| sunny_weekday | Proportional-Optimized |       0.999904 |       130903 |      1152.22 |
| sunny_weekday | Min-Cost-Flow          |       0.951673 |       125675 |         0    |
| rainy_weekend | Zero-Action            |       0.953767 |       112877 |         0    |
| rainy_weekend | Proportional-Optimized |       1        |       116869 |      1231.23 |
| rainy_weekend | Min-Cost-Flow          |       0.956998 |       112485 |         0    |
| summer_peak   | Zero-Action            |       0.940078 |       127446 |         0    |
| summer_peak   | Proportional-Optimized |       0.99988  |       134366 |      1173.68 |
| summer_peak   | Min-Cost-Flow          |       0.947512 |       128244 |         0    |
| winter_low    | Zero-Action            |       0.973888 |       110299 |         0    |
| winter_low    | Proportional-Optimized |       1        |       112705 |      1025.6  |
| winter_low    | Min-Cost-Flow          |       0.962948 |       108986 |         0    |