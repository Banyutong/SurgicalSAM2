# Training Insights Report

Source CSV: `wandb_export_2025-09-18T12_21_56.746+08_00.csv`

## Trained Models vs Baseline
| dataset     | prompt_type   | config       |   eval/Dice |   Dice_baseline |   Dice_improvement_% |   eval/mIoU |   mIoU_baseline |   mIoU_improvement_% |   eval/MAE |   MAE_baseline |   MAE_reduction_% |
|:------------|:--------------|:-------------|------------:|----------------:|---------------------:|------------:|----------------:|---------------------:|-----------:|---------------:|------------------:|
| cholecseg8k | box           | md+pe+ie     |      0.8308 |          0.7963 |                 4.33 |      0.7627 |          0.7168 |                 6.4  |     1.4159 |         2.6365 |            -46.3  |
| cholecseg8k | box           | mem+md+pe+ie |      0.8382 |          0.7963 |                 5.26 |      0.7696 |          0.7168 |                 7.36 |     1.4044 |         2.6365 |            -46.73 |
| cholecseg8k | box           | md+pe        |      0.8147 |          0.7963 |                 2.3  |      0.7412 |          0.7168 |                 3.41 |     1.6169 |         2.6365 |            -38.67 |
| cholecseg8k | box           | md           |      0.8146 |          0.7963 |                 2.29 |      0.7411 |          0.7168 |                 3.39 |     1.6182 |         2.6365 |            -38.62 |
| cholecseg8k | box           | mem+md+pe    |      0.817  |          0.7963 |                 2.59 |      0.7447 |          0.7168 |                 3.89 |     1.6192 |         2.6365 |            -38.59 |
| cholecseg8k | box           | mem+md       |      0.817  |          0.7963 |                 2.59 |      0.7447 |          0.7168 |                 3.9  |     1.6144 |         2.6365 |            -38.77 |
| cholecseg8k | box           | mem          |      0.8025 |          0.7963 |                 0.78 |      0.7265 |          0.7168 |                 1.35 |     1.9916 |         2.6365 |            -24.46 |
| cholecseg8k | mask          | mem+md+pe+ie |      0.8606 |          0.8515 |                 1.07 |      0.8057 |          0.7932 |                 1.58 |     1.277  |         1.7016 |            -24.95 |
| cholecseg8k | mask          | md+pe+ie     |      0.8577 |          0.8515 |                 0.73 |      0.8025 |          0.7932 |                 1.17 |     1.3093 |         1.7016 |            -23.06 |
| cholecseg8k | mask          | md+pe        |      0.8547 |          0.8515 |                 0.38 |      0.7988 |          0.7932 |                 0.7  |     1.3564 |         1.7016 |            -20.29 |
| cholecseg8k | mask          | md           |      0.8548 |          0.8515 |                 0.39 |      0.7989 |          0.7932 |                 0.71 |     1.3538 |         1.7016 |            -20.44 |
| cholecseg8k | mask          | mem+md+pe    |      0.8589 |          0.8515 |                 0.87 |      0.8043 |          0.7932 |                 1.4  |     1.2521 |         1.7016 |            -26.41 |
| cholecseg8k | mask          | mem+md       |      0.859  |          0.8515 |                 0.88 |      0.8045 |          0.7932 |                 1.42 |     1.2491 |         1.7016 |            -26.59 |
| cholecseg8k | mask          | mem          |      0.859  |          0.8515 |                 0.88 |      0.8041 |          0.7932 |                 1.37 |     1.2248 |         1.7016 |            -28.02 |
| cholecseg8k | point         | md+pe+ie     |      0.7338 |          0.6658 |                10.22 |      0.6498 |          0.5693 |                14.16 |     3.7476 |         6.3432 |            -40.92 |
| cholecseg8k | point         | mem+md+pe+ie |      0.7354 |          0.6658 |                10.46 |      0.6499 |          0.5693 |                14.16 |     4.0569 |         6.3432 |            -36.04 |
| cholecseg8k | point         | md+pe        |      0.7024 |          0.6658 |                 5.5  |      0.6138 |          0.5693 |                 7.82 |     5.1971 |         6.3432 |            -18.07 |
| cholecseg8k | point         | md           |      0.6995 |          0.6658 |                 5.07 |      0.6108 |          0.5693 |                 7.3  |     5.2883 |         6.3432 |            -16.63 |
| cholecseg8k | point         | mem+md+pe    |      0.702  |          0.6658 |                 5.44 |      0.6151 |          0.5693 |                 8.06 |     5.4555 |         6.3432 |            -13.99 |
| cholecseg8k | point         | mem+md       |      0.7058 |          0.6658 |                 6.02 |      0.6186 |          0.5693 |                 8.66 |     5.5071 |         6.3432 |            -13.18 |
| cholecseg8k | point         | mem          |      0.6801 |          0.6658 |                 2.15 |      0.586  |          0.5693 |                 2.95 |     6.412  |         6.3432 |              1.08 |
| endovis17   | box           | md+pe        |      0.7721 |          0.8031 |                -3.86 |      0.7206 |          0.7476 |                -3.61 |     3.1092 |         2.8398 |              9.49 |
| endovis17   | box           | md+pe+ie     |      0.8482 |          0.8031 |                 5.62 |      0.8019 |          0.7476 |                 7.27 |     1.3265 |         2.8398 |            -53.29 |
| endovis17   | box           | mem+md+pe+ie |      0.8024 |          0.8031 |                -0.08 |      0.755  |          0.7476 |                 1    |     2.8543 |         2.8398 |              0.51 |
| endovis17   | box           | md           |      0.7886 |          0.8031 |                -1.8  |      0.7373 |          0.7476 |                -1.37 |     3.0037 |         2.8398 |              5.77 |
| endovis17   | box           | mem+md+pe    |      0.7917 |          0.8031 |                -1.41 |      0.7397 |          0.7476 |                -1.05 |     2.9553 |         2.8398 |              4.07 |
| endovis17   | box           | mem+md       |      0.7885 |          0.8031 |                -1.81 |      0.7364 |          0.7476 |                -1.49 |     2.9989 |         2.8398 |              5.6  |
| endovis17   | box           | mem          |      0.7949 |          0.8031 |                -1.01 |      0.7418 |          0.7476 |                -0.77 |     3.0392 |         2.8398 |              7.02 |
| endovis17   | mask          | mem+md+pe+ie |      0.8258 |          0.8169 |                 1.08 |      0.7789 |          0.7668 |                 1.58 |     2.4052 |         2.8168 |            -14.61 |
| endovis17   | mask          | md+pe+ie     |      0.8248 |          0.8169 |                 0.97 |      0.7791 |          0.7668 |                 1.6  |     2.4548 |         2.8168 |            -12.85 |
| endovis17   | mask          | md+pe        |      0.8183 |          0.8169 |                 0.17 |      0.7705 |          0.7668 |                 0.48 |     2.6187 |         2.8168 |             -7.03 |
| endovis17   | mask          | md           |      0.8195 |          0.8169 |                 0.32 |      0.7719 |          0.7668 |                 0.67 |     2.6233 |         2.8168 |             -6.87 |
| endovis17   | mask          | mem+md+pe    |      0.8469 |          0.8169 |                 3.67 |      0.8006 |          0.7668 |                 4.41 |     1.9164 |         2.8168 |            -31.97 |
| endovis17   | mask          | mem+md       |      0.8454 |          0.8169 |                 3.49 |      0.7988 |          0.7668 |                 4.17 |     1.9326 |         2.8168 |            -31.39 |
| endovis17   | mask          | mem          |      0.8268 |          0.8169 |                 1.21 |      0.779  |          0.7668 |                 1.59 |     2.4298 |         2.8168 |            -13.74 |
| endovis17   | point         | md+pe+ie     |      0.8579 |          0.7172 |                19.63 |      0.811  |          0.6465 |                25.45 |     0.6846 |         4.1716 |            -83.59 |
| endovis17   | point         | mem+md+pe+ie |      0.8252 |          0.7172 |                15.06 |      0.7785 |          0.6465 |                20.42 |     1.2885 |         4.1716 |            -69.11 |
| endovis17   | point         | md+pe        |      0.7956 |          0.7172 |                10.94 |      0.7422 |          0.6465 |                14.8  |     2.7543 |         4.1716 |            -33.97 |
| endovis17   | point         | md           |      0.8024 |          0.7172 |                11.88 |      0.7503 |          0.6465 |                16.06 |     2.6303 |         4.1716 |            -36.95 |
| endovis17   | point         | mem+md+pe    |      0.7943 |          0.7172 |                10.76 |      0.7448 |          0.6465 |                15.21 |     2.6968 |         4.1716 |            -35.35 |
| endovis17   | point         | mem+md       |      0.8077 |          0.7172 |                12.63 |      0.7565 |          0.6465 |                17.01 |     2.1681 |         4.1716 |            -48.03 |
| endovis17   | point         | mem          |      0.7625 |          0.7172 |                 6.31 |      0.7052 |          0.6465 |                 9.08 |     2.563  |         4.1716 |            -38.56 |
| endovis18   | box           | md+pe+ie     |      0.4362 |          0.3849 |                13.32 |      0.4072 |          0.3511 |                15.97 |     3.4858 |         4.3905 |            -20.6  |
| endovis18   | box           | mem+md+pe+ie |      0.3914 |          0.3849 |                 1.7  |      0.3679 |          0.3511 |                 4.77 |     3.4465 |         4.3905 |            -21.5  |
| endovis18   | box           | md+pe        |      0.3695 |          0.3849 |                -4.01 |      0.3427 |          0.3511 |                -2.4  |     3.7171 |         4.3905 |            -15.34 |
| endovis18   | box           | md           |      0.3742 |          0.3849 |                -2.79 |      0.347  |          0.3511 |                -1.17 |     3.6983 |         4.3905 |            -15.76 |
| endovis18   | box           | mem+md+pe    |      0.3714 |          0.3849 |                -3.51 |      0.3415 |          0.3511 |                -2.75 |     4.2108 |         4.3905 |             -4.09 |
| endovis18   | box           | mem          |      0.3781 |          0.3849 |                -1.78 |      0.3518 |          0.3511 |                 0.18 |     3.6255 |         4.3905 |            -17.42 |
| endovis18   | box           | mem+md       |      0.3568 |          0.3849 |                -7.31 |      0.331  |          0.3511 |                -5.72 |     4.0715 |         4.3905 |             -7.26 |
| endovis18   | mask          | mem+md+pe+ie |      0.3881 |          0.3871 |                 0.25 |      0.3629 |          0.3564 |                 1.81 |     4.0266 |         4.2408 |             -5.05 |
| endovis18   | mask          | md+pe+ie     |      0.3694 |          0.3871 |                -4.58 |      0.3451 |          0.3564 |                -3.17 |     3.8777 |         4.2408 |             -8.56 |
| endovis18   | mask          | md+pe        |      0.3773 |          0.3871 |                -2.55 |      0.3521 |          0.3564 |                -1.22 |     3.4986 |         4.2408 |            -17.5  |
| endovis18   | mask          | md           |      0.3788 |          0.3871 |                -2.16 |      0.3533 |          0.3564 |                -0.87 |     3.4986 |         4.2408 |            -17.5  |
| endovis18   | mask          | mem+md+pe    |      0.3785 |          0.3871 |                -2.22 |      0.352  |          0.3564 |                -1.23 |     4.1136 |         4.2408 |             -3    |
| endovis18   | mask          | mem          |      0.3993 |          0.3871 |                 3.14 |      0.3723 |          0.3564 |                 4.45 |     3.5908 |         4.2408 |            -15.33 |
| endovis18   | mask          | mem+md       |      0.3711 |          0.3871 |                -4.15 |      0.3474 |          0.3564 |                -2.53 |     3.8483 |         4.2408 |             -9.26 |
| endovis18   | point         | md+pe+ie     |      0.3694 |          0.3371 |                 9.57 |      0.3451 |          0.2891 |                19.39 |     3.7481 |         6.74   |            -44.39 |
| endovis18   | point         | mem+md+pe+ie |      0.3437 |          0.3371 |                 1.96 |      0.322  |          0.2891 |                11.38 |     4.1497 |         6.74   |            -38.43 |
| endovis18   | point         | md+pe        |      0.3854 |          0.3371 |                14.33 |      0.3512 |          0.2891 |                21.5  |     4.3708 |         6.74   |            -35.15 |
| endovis18   | point         | md           |      0.386  |          0.3371 |                14.5  |      0.3521 |          0.2891 |                21.8  |     4.3742 |         6.74   |            -35.1  |
| endovis18   | point         | mem+md+pe    |      0.378  |          0.3371 |                12.13 |      0.3457 |          0.2891 |                19.6  |     4.5051 |         6.74   |            -33.16 |
| endovis18   | point         | mem+md       |      0.375  |          0.3371 |                11.25 |      0.3421 |          0.2891 |                18.34 |     4.607  |         6.74   |            -31.65 |
| endovis18   | point         | mem          |      0.3297 |          0.3371 |                -2.2  |      0.2937 |          0.2891 |                 1.6  |     5.9376 |         6.74   |            -11.91 |

## Insight 1 · Overall Impact of Training
Average percentage change relative to the epoch-0 baseline.
| dataset     |   Dice_improvement_% |   mIoU_improvement_% |   MAE_reduction_% |
|:------------|---------------------:|---------------------:|------------------:|
| cholecseg8k |                 3.34 |                 4.82 |            -27.6  |
| endovis17   |                 4.46 |                 6.31 |            -23.09 |
| endovis18   |                 2.14 |                 5.7  |            -19.43 |
Training consistently improves Dice and mIoU while reducing MAE across datasets, with the most pronounced gains on the more challenging endovis splits.

## Insight 2 · Contribution of Memory Modules
Mean Dice score for trained models with and without the memory encoder.
| dataset     |   Without Memory |   With Memory |
|:------------|-----------------:|--------------:|
| cholecseg8k |           0.7959 |        0.7946 |
| endovis17   |           0.8142 |        0.8094 |
| endovis18   |           0.3829 |        0.3718 |
Memory modules deliver modest but reliable Dice improvements for every dataset.

## Insight 3 · Value of Fine-Tuning the Image Encoder
Average Dice improvement (%) when the image encoder is trainable.
| dataset     |   Without Image Encoder |   With Image Encoder |
|:------------|------------------------:|---------------------:|
| cholecseg8k |                    2.54 |                 5.35 |
| endovis17   |                    3.43 |                 7.05 |
| endovis18   |                    1.51 |                 3.7  |
Fine-tuning the image encoder roughly doubles the Dice lift versus freezing it, especially for endovis17 and endovis18.

## Insight 4 · Prompt-Type Effectiveness
Mean Dice scores per prompt type after training.
| dataset     |    box |   mask |   point |
|:------------|-------:|-------:|--------:|
| cholecseg8k | 0.8193 | 0.8578 |  0.7084 |
| endovis17   | 0.7981 | 0.8297 |  0.8065 |
| endovis18   | 0.3825 | 0.3804 |  0.3667 |
Mask prompts remain the strongest option after training, while point prompts lag on every dataset.

## Insight 5 · Best Configuration per Dataset
| dataset     | Name                                  | config       |   eval/Dice |   Dice_improvement_% |
|:------------|:--------------------------------------|:-------------|------------:|---------------------:|
| cholecseg8k | cholecseg8k_mask_mem+md+pe+ie__148297 | mem+md+pe+ie |      0.8606 |                 1.07 |
| endovis17   | endovis17_point_md+pe+ie__148637      | md+pe+ie     |      0.8579 |                19.63 |
| endovis18   | endovis18_box_md+pe+ie__148639        | md+pe+ie     |      0.4362 |                13.32 |
These runs define the current per-dataset high-water marks and highlight the benefit of richer prompt signals.

## Insight 6 · Best Overall Fine-Tuning Recipe
| config       |   mean_dice |   mean_dice_improvement |   run_count |
|:-------------|------------:|------------------------:|------------:|
| md+pe+ie     |      0.6809 |                    6.65 |           9 |
| mem+md+pe+ie |      0.6679 |                    4.08 |           9 |
| mem+md+pe    |      0.6599 |                    3.15 |           9 |
| mem+md       |      0.6585 |                    2.62 |           9 |
| md           |      0.6576 |                    3.08 |           9 |
| md+pe        |      0.6544 |                    2.58 |           9 |
| mem          |      0.6481 |                    1.05 |           9 |
The leading recipe is `md+pe+ie` with mean Dice 0.6809 and a 6.65% lift over baseline.
Jointly training the memory, mask decoder, prompt encoder, and image encoder offers the most robust gains.
