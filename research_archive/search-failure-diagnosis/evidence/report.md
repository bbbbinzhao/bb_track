# 搜索裁剪失败、SRA 响应与缩小裁剪诊断

完整官方 val：90 条序列；复用已审计的原 SRA 轨迹，使用原始 exist/GT 重新计算。初始化帧排除。

当前帧失败 IoU<0.2；完整覆盖为框覆盖率≥0.999；连续五帧失败触发失效事件，连续三帧 IoU≥0.5 确认恢复。恢复观察在目标再次消失、GT 非法或序列结束时截断。

SRA 同时要求 presence<0.8 和原始峰值<0.5；触发不等于实际扩大，因子可能已到 8。因子上升也不保证实际像素边长上升，因为预测框始终更新。

## 完整统计

```json
{
  "valid_frames": 92460,
  "low_iou_frames": 14129,
  "low_iou_category_counts": {
    "outside_center": 12840,
    "partly_covered": 164,
    "fully_covered": 1125
  },
  "covered_low_frames": 1125,
  "covered_low_95percent": 1128,
  "covered_low_binary_presence_correct_at_05": 717,
  "covered_low_binary_presence_wrong_at_05": 408,
  "covered_low_presence_between_05_and_08": 251,
  "covered_low_signal_table": {
    "presence_high_peak_high": 114,
    "presence_low_peak_high": 171,
    "presence_low_peak_low": 488,
    "presence_high_peak_low": 352
  },
  "outside_low_signal_table": {
    "presence_low_peak_high": 3966,
    "presence_low_peak_low": 3131,
    "presence_high_peak_low": 2256,
    "presence_high_peak_high": 3487
  },
  "crop_exit_events": {
    "events": 266,
    "sequences": 43,
    "recovery_rate_30": 0.556390977443609,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 19,
    "recovered_before_censoring": 202,
    "recovery_confirmation_mean_success_only": 36.277227722772274,
    "recovery_confirmation_median_success_only": 5.5,
    "trigger_first_frame_rate": 0.4473684210526316,
    "never_triggered_before_recovery_or_censoring": 81,
    "actual_expansion_first_frame_rate": 0.3233082706766917,
    "first_trigger_mean_detected_only": 7.032432432432432,
    "first_full_coverage_mean_observed_only": 25.56387665198238,
    "ever_full_coverage": 227,
    "fully_covered_at_least_once_but_not_recovered": 25,
    "coverage_to_first_good_mean_observed_only": 21.52112676056338,
    "coverage_to_recovery_mean_recovered_only": 21.361386138613863,
    "covered_but_low_events": 93
  },
  "failed_crop_exits_with_valid_previous_frame": {
    "events": 232,
    "sequences": 35,
    "recovery_rate_30": 0.5862068965517241,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 13,
    "recovered_before_censoring": 186,
    "recovery_confirmation_mean_success_only": 33.16129032258065,
    "recovery_confirmation_median_success_only": 5.0,
    "trigger_first_frame_rate": 0.46120689655172414,
    "never_triggered_before_recovery_or_censoring": 72,
    "actual_expansion_first_frame_rate": 0.3448275862068966,
    "first_trigger_mean_detected_only": 7.88125,
    "first_full_coverage_mean_observed_only": 22.751196172248804,
    "ever_full_coverage": 209,
    "fully_covered_at_least_once_but_not_recovered": 23,
    "coverage_to_first_good_mean_observed_only": 21.739795918367346,
    "coverage_to_recovery_mean_recovered_only": 22.188172043010752,
    "covered_but_low_events": 86
  },
  "failed_crop_exits_previous_good": {
    "events": 87,
    "sequences": 17,
    "recovery_rate_30": 0.9540229885057471,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 1,
    "recovered_before_censoring": 86,
    "recovery_confirmation_mean_success_only": 7.267441860465116,
    "recovery_confirmation_median_success_only": 4.0,
    "trigger_first_frame_rate": 0.40229885057471265,
    "never_triggered_before_recovery_or_censoring": 47,
    "actual_expansion_first_frame_rate": 0.40229885057471265,
    "first_trigger_mean_detected_only": 0.95,
    "first_full_coverage_mean_observed_only": 2.9195402298850577,
    "ever_full_coverage": 87,
    "fully_covered_at_least_once_but_not_recovered": 1,
    "coverage_to_first_good_mean_observed_only": 0.28735632183908044,
    "coverage_to_recovery_mean_recovered_only": 4.337209302325581,
    "covered_but_low_events": 4
  },
  "failed_crop_exits_previous_bad": {
    "events": 118,
    "sequences": 22,
    "recovery_rate_30": 0.2288135593220339,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 12,
    "recovered_before_censoring": 73,
    "recovery_confirmation_mean_success_only": 73.42465753424658,
    "recovery_confirmation_median_success_only": 43.0,
    "trigger_first_frame_rate": 0.5169491525423728,
    "never_triggered_before_recovery_or_censoring": 12,
    "actual_expansion_first_frame_rate": 0.288135593220339,
    "first_trigger_mean_detected_only": 11.481132075471699,
    "first_full_coverage_mean_observed_only": 46.65263157894737,
    "ever_full_coverage": 95,
    "fully_covered_at_least_once_but_not_recovered": 22,
    "coverage_to_first_good_mean_observed_only": 51.02439024390244,
    "coverage_to_recovery_mean_recovered_only": 49.863013698630134,
    "covered_but_low_events": 79
  },
  "failed_crop_exits_previous_good_gt_previous_box_still_misses": 72,
  "failed_crop_exits_previous_bad_gt_previous_center_covers": 116,
  "crop_exit_previous_good": {
    "events": 89,
    "sequences": 17,
    "recovery_rate_30": 0.9550561797752809,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 1,
    "recovered_before_censoring": 88,
    "recovery_confirmation_mean_success_only": 7.193181818181818,
    "recovery_confirmation_median_success_only": 4.0,
    "trigger_first_frame_rate": 0.39325842696629215,
    "never_triggered_before_recovery_or_censoring": 47,
    "actual_expansion_first_frame_rate": 0.39325842696629215,
    "first_trigger_mean_detected_only": 0.9523809523809523,
    "first_full_coverage_mean_observed_only": 2.898876404494382,
    "ever_full_coverage": 89,
    "fully_covered_at_least_once_but_not_recovered": 1,
    "coverage_to_first_good_mean_observed_only": 0.2808988764044944,
    "coverage_to_recovery_mean_recovered_only": 4.284090909090909,
    "covered_but_low_events": 4
  },
  "crop_exit_previous_bad": {
    "events": 118,
    "sequences": 22,
    "recovery_rate_30": 0.2288135593220339,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 12,
    "recovered_before_censoring": 73,
    "recovery_confirmation_mean_success_only": 73.42465753424658,
    "recovery_confirmation_median_success_only": 43.0,
    "trigger_first_frame_rate": 0.5169491525423728,
    "never_triggered_before_recovery_or_censoring": 12,
    "actual_expansion_first_frame_rate": 0.288135593220339,
    "first_trigger_mean_detected_only": 11.481132075471699,
    "first_full_coverage_mean_observed_only": 46.65263157894737,
    "ever_full_coverage": 95,
    "fully_covered_at_least_once_but_not_recovered": 22,
    "coverage_to_first_good_mean_observed_only": 51.02439024390244,
    "coverage_to_recovery_mean_recovered_only": 49.863013698630134,
    "covered_but_low_events": 79
  },
  "crop_exit_previous_missing": 32,
  "previous_bad_exits_corrected_previous_center_covers": 116,
  "previous_good_exits_corrected_previous_box_still_misses": 72,
  "covered_low_previous5_gt_reliable": 28,
  "covered_low_previous5_online_gate": 28,
  "sustained_losses": {
    "events": 91,
    "sequences": 36,
    "recovery_rate_30": 0.42857142857142855,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 5,
    "recovered_before_censoring": 61,
    "recovery_confirmation_mean_success_only": 59.75409836065574,
    "recovery_confirmation_median_success_only": 18.0,
    "trigger_first_frame_rate": 0.3626373626373626,
    "never_triggered_before_recovery_or_censoring": 21,
    "actual_expansion_first_frame_rate": 0.21978021978021978,
    "first_trigger_mean_detected_only": 10.628571428571428,
    "first_full_coverage_mean_observed_only": 18.31168831168831,
    "ever_full_coverage": 77,
    "fully_covered_at_least_once_but_not_recovered": 16,
    "coverage_to_first_good_mean_observed_only": 33.96825396825397,
    "coverage_to_recovery_mean_recovered_only": 37.885245901639344,
    "covered_but_low_events": 62
  },
  "covered_sustained_losses": {
    "events": 48,
    "sequences": 20,
    "recovery_rate_30": 0.4583333333333333,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 1,
    "recovered_before_censoring": 34,
    "recovery_confirmation_mean_success_only": 55.14705882352941,
    "recovery_confirmation_median_success_only": 17.5,
    "trigger_first_frame_rate": 0.375,
    "never_triggered_before_recovery_or_censoring": 9,
    "actual_expansion_first_frame_rate": 0.20833333333333334,
    "first_trigger_mean_detected_only": 16.58974358974359,
    "first_full_coverage_mean_observed_only": 0.0,
    "ever_full_coverage": 48,
    "fully_covered_at_least_once_but_not_recovered": 14,
    "coverage_to_first_good_mean_observed_only": 50.2,
    "coverage_to_recovery_mean_recovered_only": 55.14705882352941,
    "covered_but_low_events": 48
  },
  "covered_sustained_losses_both_signals_low": {
    "events": 18,
    "sequences": 12,
    "recovery_rate_30": 0.2222222222222222,
    "rate30_is_observed_lower_bound_if_early_censored": true,
    "unrecovered_censored_before_30": 0,
    "recovered_before_censoring": 13,
    "recovery_confirmation_mean_success_only": 116.6923076923077,
    "recovery_confirmation_median_success_only": 79.0,
    "trigger_first_frame_rate": 1.0,
    "never_triggered_before_recovery_or_censoring": 0,
    "actual_expansion_first_frame_rate": 0.5555555555555556,
    "first_trigger_mean_detected_only": 0.0,
    "first_full_coverage_mean_observed_only": 0.0,
    "ever_full_coverage": 18,
    "fully_covered_at_least_once_but_not_recovered": 5,
    "coverage_to_first_good_mean_observed_only": 110.23076923076923,
    "coverage_to_recovery_mean_recovered_only": 116.6923076923077,
    "covered_but_low_events": 18
  },
  "attribute_associations_in_available_annotations": {
    "OC=0": {
      "frames": 24083,
      "sequences": 17,
      "low_iou_rate": 0.4582070340073911,
      "covered_low_rate": 0.03923929742972221
    },
    "OC=1": {
      "frames": 3,
      "sequences": 2,
      "low_iou_rate": 0.6666666666666666,
      "covered_low_rate": 0.0
    },
    "FM=0": {
      "frames": 21137,
      "sequences": 17,
      "low_iou_rate": 0.4507262146946113,
      "covered_low_rate": 0.03713866679282774
    },
    "FM=1": {
      "frames": 2949,
      "sequences": 11,
      "low_iou_rate": 0.5120379789759241,
      "covered_low_rate": 0.05425567989148864
    },
    "TS=0": {
      "frames": 9813,
      "sequences": 16,
      "low_iou_rate": 0.5380617548150413,
      "covered_low_rate": 0.07469683073473964
    },
    "TS=1": {
      "frames": 14077,
      "sequences": 17,
      "low_iou_rate": 0.4081125239752788,
      "covered_low_rate": 0.014988989131206934
    },
    "LR=0": {
      "frames": 24086,
      "sequences": 17,
      "low_iou_rate": 0.45823299842232,
      "covered_low_rate": 0.03923441003072324
    },
    "LR=1": {
      "frames": 0,
      "sequences": 0,
      "low_iou_rate": null,
      "covered_low_rate": null
    },
    "IC=0": {
      "frames": 14789,
      "sequences": 14,
      "low_iou_rate": 0.1860842518087768,
      "covered_low_rate": 0.01088646967340591
    },
    "IC=1": {
      "frames": 9297,
      "sequences": 15,
      "low_iou_rate": 0.8911476820479725,
      "covered_low_rate": 0.08432827793912015
    },
    "DBC=0": {
      "frames": 12605,
      "sequences": 12,
      "low_iou_rate": 0.31868306227687426,
      "covered_low_rate": 0.032130107100357
    },
    "DBC=1": {
      "frames": 11481,
      "sequences": 16,
      "low_iou_rate": 0.6114449960804808,
      "covered_low_rate": 0.04703423046772929
    }
  },
  "shrink": {
    "all_covered_failures": 1125,
    "feasible_shrink": 779,
    "oracle_gain_005": 225,
    "oracle_success_iou05": 87,
    "oracle_success_sequences": 14,
    "oracle_success_previous5_gt_reliable": 10,
    "oracle_success_previous5_online_gate": 1,
    "oracle_success_both_histories": 1,
    "oracle_success_previous5_gt_reliable_sequences": [
      "20190925_133630_1_9",
      "20190925_141417_1_1",
      "3700000000002_105103_3",
      "3700000000002_105103_8",
      "3700000000002_110743_1",
      "3700000000002_111356_3"
    ],
    "oracle_success_previous5_online_gate_sequences": [
      "3700000000002_110743_1"
    ],
    "covered_failure_previous5_gt_reliable_total": 28,
    "covered_failure_previous5_online_gate_total": 28,
    "history_window_sensitivity_descriptive": {
      "1": {
        "covered_failures_gt_reliable": 56,
        "rescued_gt_reliable": 22,
        "covered_failures_both_confident": 121,
        "rescued_both_confident": 5
      },
      "3": {
        "covered_failures_gt_reliable": 36,
        "rescued_gt_reliable": 17,
        "covered_failures_both_confident": 57,
        "rescued_both_confident": 2
      },
      "5": {
        "covered_failures_gt_reliable": 28,
        "rescued_gt_reliable": 10,
        "covered_failures_both_confident": 34,
        "rescued_both_confident": 1
      }
    },
    "uniform_and_gate_sample_actions": {
      "uniform_factor2.0": {
        "frames": 2831,
        "sequence_paired_delta": {
          "sequences": 90,
          "mean": -0.04629253385888883,
          "ci95": [
            -0.05442983346901254,
            -0.03808014330537727
          ]
        },
        "harm_ge005_rate": 0.3754856941010244,
        "gain_ge005_rate": 0.07417873542917697,
        "coverage_lost_rate": 0.08689509007417874,
        "good_to_low_iou_rate": 0.015542211232779936,
        "both_crops_fully_cover": {
          "frames": 2296,
          "sequence_paired_delta": {
            "sequences": 90,
            "mean": -0.03334974853082995,
            "ci95": [
              -0.04246827125999937,
              -0.024130151750114782
            ]
          },
          "harm_ge005_rate": 0.382404181184669
        }
      },
      "uniform_factor3.0": {
        "frames": 2831,
        "sequence_paired_delta": {
          "sequences": 90,
          "mean": -0.016550026679764306,
          "ci95": [
            -0.021648144412695216,
            -0.01164951467907278
          ]
        },
        "harm_ge005_rate": 0.15718827269516072,
        "gain_ge005_rate": 0.06358177322500883,
        "coverage_lost_rate": 0.026139173436948077,
        "good_to_low_iou_rate": 0.007417873542917697,
        "both_crops_fully_cover": {
          "frames": 2468,
          "sequence_paired_delta": {
            "sequences": 90,
            "mean": -0.009197636323050592,
            "ci95": [
              -0.014978509601375194,
              -0.00348860838925959
            ]
          },
          "harm_ge005_rate": 0.15964343598055106
        }
      },
      "uniform_factor4.0": {
        "frames": 2831,
        "sequence_paired_delta": {
          "sequences": 90,
          "mean": -0.007779497981478861,
          "ci95": [
            -0.011707357973367276,
            -0.004008267187478904
          ]
        },
        "harm_ge005_rate": 0.0738255033557047,
        "gain_ge005_rate": 0.04839279406570116,
        "coverage_lost_rate": 0.01377605086541858,
        "good_to_low_iou_rate": 0.004945249028611798,
        "both_crops_fully_cover": {
          "frames": 2503,
          "sequence_paired_delta": {
            "sequences": 90,
            "mean": -0.0033776042787000025,
            "ci95": [
              -0.007084606510162302,
              0.00038868014309551864
            ]
          },
          "harm_ge005_rate": 0.07271274470635238
        }
      },
      "uniform_factor5.0": {
        "frames": 2831,
        "sequence_paired_delta": {
          "sequences": 90,
          "mean": -0.003364053137298399,
          "ci95": [
            -0.0057661515695751845,
            -0.0011069176880450292
          ]
        },
        "harm_ge005_rate": 0.028965030024726245,
        "gain_ge005_rate": 0.02190038855528082,
        "coverage_lost_rate": 0.006711409395973154,
        "good_to_low_iou_rate": 0.002472624514305899,
        "both_crops_fully_cover": {
          "frames": 2523,
          "sequence_paired_delta": {
            "sequences": 90,
            "mean": -0.00014523925251576044,
            "ci95": [
              -0.0030596141837732982,
              0.003263373835562502
            ]
          },
          "harm_ge005_rate": 0.026159334126040427
        }
      },
      "online_gate_sample_factor2.0": {
        "frames": 2740,
        "sequence_paired_delta": {
          "sequences": 88,
          "mean": -0.041351227455178614,
          "ci95": [
            -0.049841912255961206,
            -0.03273641547151327
          ]
        },
        "harm_ge005_rate": 0.3886861313868613,
        "gain_ge005_rate": 0.08576642335766424,
        "coverage_lost_rate": 0.0864963503649635,
        "good_to_low_iou_rate": 0.00583941605839416,
        "both_crops_fully_cover": {
          "frames": 2381,
          "sequence_paired_delta": {
            "sequences": 87,
            "mean": -0.031535730714165225,
            "ci95": [
              -0.041480326838545335,
              -0.02167425031225402
            ]
          },
          "harm_ge005_rate": 0.36875262494750105
        }
      },
      "online_gate_sample_factor3.0": {
        "frames": 2740,
        "sequence_paired_delta": {
          "sequences": 88,
          "mean": -0.011461691401411745,
          "ci95": [
            -0.01627566229068292,
            -0.006614404476762383
          ]
        },
        "harm_ge005_rate": 0.15693430656934307,
        "gain_ge005_rate": 0.06897810218978102,
        "coverage_lost_rate": 0.01167883211678832,
        "good_to_low_iou_rate": 0.00072992700729927,
        "both_crops_fully_cover": {
          "frames": 2586,
          "sequence_paired_delta": {
            "sequences": 88,
            "mean": -0.0083546835597958,
            "ci95": [
              -0.013421380420665796,
              -0.0031571015042538998
            ]
          },
          "harm_ge005_rate": 0.15622583139984533
        }
      },
      "online_gate_sample_factor4.0": {
        "frames": 2740,
        "sequence_paired_delta": {
          "sequences": 88,
          "mean": -0.004053974022932626,
          "ci95": [
            -0.007174185025173355,
            -0.0009149555938824383
          ]
        },
        "harm_ge005_rate": 0.07591240875912408,
        "gain_ge005_rate": 0.04817518248175182,
        "coverage_lost_rate": 0.0032846715328467154,
        "good_to_low_iou_rate": 0.0,
        "both_crops_fully_cover": {
          "frames": 2609,
          "sequence_paired_delta": {
            "sequences": 88,
            "mean": -0.003456004621402599,
            "ci95": [
              -0.006732893482823774,
              -0.00019953905520510333
            ]
          },
          "harm_ge005_rate": 0.07665772326561901
        }
      },
      "online_gate_sample_factor5.0": {
        "frames": 2740,
        "sequence_paired_delta": {
          "sequences": 88,
          "mean": -0.0013243786625496258,
          "ci95": [
            -0.0033326320592462085,
            0.0005416528401135381
          ]
        },
        "harm_ge005_rate": 0.02846715328467153,
        "gain_ge005_rate": 0.025547445255474453,
        "coverage_lost_rate": 0.00072992700729927,
        "good_to_low_iou_rate": 0.0,
        "both_crops_fully_cover": {
          "frames": 2616,
          "sequence_paired_delta": {
            "sequences": 88,
            "mean": -0.0024581944926529934,
            "ci95": [
              -0.007191999173417097,
              0.0009151346863993983
            ]
          },
          "harm_ge005_rate": 0.0290519877675841
        }
      }
    }
  }
}
```

## 解释边界

- 裁剪出界起点的前一帧 IoU 与 GT 位移用于分解误差；GT 修正裁剪是离线几何参照，不是在线策略。前帧 IoU 高也不等于中心、尺度完全正确。
- 图像坐标位移不能单独区分目标运动与相机运动。遮挡等属性仅在存在标注的帧上分析，属性与失败共现不证明因果。
- 中心出界事件中含未发生定位失败、以及重新出现的情况；针对问题一应优先看 failed_crop_exits_with_valid_previous_frame 及其前帧好/坏分层。事件可能来自相同失效轨迹，不能作为独立序列样本。
- 存在概率高说明分支判定搜索图有目标，不证明选中了正确目标；低于 0.8 是控制阈值，不等于二分类 argmax 错误。
- 缩小实验保持原预测中心、模板、原坐标还原与当前 Hann 状态。每个分支先复原上一帧状态；原裁剪逐任务复现、A/B/A、冻结参数和缓冲哈希均检查。
- GT 选择缩小后仍完整覆盖且 IoU 最高的候选为 oracle。缩小同时改变背景干扰和目标像素大小，不能据此单独归因于分辨率。
- 前五帧 GT IoU≥0.5 是事后可靠性；在线 gate 仅用截至前一帧的置信度和框变化。在线 gate 通过也可能稳定跟错。
- 正常帧对照按每序列最多 32 个均匀时刻和最多 32 个 gate 时刻取样，分开报告；失效帧全量枚举。不能将事件样本收益当作整体指标或闭环收益。
