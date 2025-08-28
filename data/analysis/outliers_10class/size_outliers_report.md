# Size Outliers Analysis Report (10 Classes)

## Overview
Analysis of bounding box size outliers across all 10 BDD100K classes.

## Outliers by Class

| Class | IQR Outliers | Z-score Outliers | AR Outliers | IQR % | Mean Area | Std Area |
|-------|--------------|------------------|-------------|-------|-----------|----------|
| traffic light | 19049 | 528 | 39 | 8.94% | 506 | 1608 |
| traffic sign | 29969 | 2456 | 134 | 10.91% | 1198 | 4180 |
| car | 118569 | 20804 | 189 | 14.54% | 9418 | 24997 |
| pedestrian | 11002 | 1631 | 79 | 10.52% | 2937 | 7581 |
| bus | 1947 | 435 | 4 | 14.67% | 35550 | 73017 |
| truck | 4837 | 1021 | 14 | 14.14% | 27728 | 59094 |
| rider | 629 | 117 | 2 | 12.18% | 6271 | 14121 |
| bicycle | 868 | 176 | 4 | 10.56% | 5863 | 10504 |
| motorcycle | 389 | 70 | 0 | 11.26% | 7612 | 16371 |
| train | 21 | 3 | 9 | 13.91% | 37708 | 69494 |

## Extreme Cases

- **67c7bea2-283d3be6.jpg** (train): traffic light
  - Area: 302,654 pixels²
  - Dimensions: 558 x 543
  - Aspect Ratio: 1.03

- **8edb33ee-ec1bc2e0.jpg** (train): traffic light
  - Area: 294,183 pixels²
  - Dimensions: 451 x 652
  - Aspect Ratio: 0.69

- **88089880-374b8b28.jpg** (train): traffic light
  - Area: 235,444 pixels²
  - Dimensions: 395 x 596
  - Aspect Ratio: 0.66

- **52ba66d6-d5b38e9f.jpg** (train): traffic light
  - Area: 220,496 pixels²
  - Dimensions: 451 x 488
  - Aspect Ratio: 0.92

- **5ccd45ac-5b127b2d.jpg** (train): traffic light
  - Area: 167,812 pixels²
  - Dimensions: 293 x 572
  - Aspect Ratio: 0.51

- **1d33c83b-71e1ea1c.jpg** (train): traffic sign
  - Area: 917,710 pixels²
  - Dimensions: 1279 x 717
  - Aspect Ratio: 1.78

- **2e5a3ced-c7603d0d.jpg** (train): traffic sign
  - Area: 906,063 pixels²
  - Dimensions: 1274 x 711
  - Aspect Ratio: 1.79

- **1af55d81-20ae3997.jpg** (train): traffic sign
  - Area: 366,158 pixels²
  - Dimensions: 793 x 462
  - Aspect Ratio: 1.72

- **14e578d7-c17a7d22.jpg** (train): traffic sign
  - Area: 323,745 pixels²
  - Dimensions: 561 x 577
  - Aspect Ratio: 0.97

- **39fba3aa-43b709f1.jpg** (train): traffic sign
  - Area: 299,441 pixels²
  - Dimensions: 788 x 380
  - Aspect Ratio: 2.08

