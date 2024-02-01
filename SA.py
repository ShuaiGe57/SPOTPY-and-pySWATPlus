import os
import pandas as pd
import numpy as np
import spotpy as sp
import shutil
import mpi4py
import sys
import multiprocessing

from pySWATPlus.TxtinoutReader import TxtinoutReader
from pySWATPlus.FileReader import FileReader
from matplotlib import pyplot as plt
from datetime import datetime
from SALib.sample.morris import morris as sample
from SALib.analyze import morris as analyze
from dask.distributed import LocalCluster


def parallel_swat(params_list):
    # 这里是不是只运行一组参数就行
    cwd, reader, swat_params, copy_path, show_output, delete_copy = params_list
    result = reader.copy_and_run(dir=copy_path,
                                 params=swat_params,
                                 show_output=show_output
                                 )
    # todo 路径直接输入，不要用os拼
    reader = FileReader(os.path.join(result, "basin_aqu_mon.txt"),
                        has_units=True,
                        index=None,
                        usecols=["mon", "day", "yr", "unit", "no3_lat"],
                        filter_by={"unit": 1}
                        )
    res = reader.df
    res["Date"] = pd.to_datetime(pd.DataFrame({"year": res["yr"],
                                               "month": res["mon"],
                                               "day": res["day"]}))
    # res.drop(columns=["Date", "mon", "day", "yr", "unit"], inplace=True)
    # todo deprecated
    os.chdir(cwd)  # 改回当前路径

    return res["no3_lat"].mean()


def save_list_to_file(lst, filename):
    with open(filename, "w") as file:
        for item in lst:
            file.write(str(item) + "\n")


if __name__ == "__main__":
    problem = {
        "num_vars": 46,
        "names": [
            "lat_ttime",
            "can_max",
            "esco",
            "epco",
            "cn3_swf",
            "perco",
            "pet_co",
            "latq_co",
            "cn_a",
            "cn_b",
            "cn_c",
            "cn_d",
            "gw_flo",
            "dep_bot",
            "dep_wt",
            "flo_dist",
            "bf_max",
            "alpha_bf",
            "revap",
            "rchg_dp",
            "spec_yld",
            "flo_min",
            "revap_min",
            "fall_tmp",
            "melt_tmp",
            "melt_max_min",
            "melt_lag",
            "snow_h2o",
            "cov50",
            "dp",
            "t_fc",
            "lag",
            "rad",
            "dist",
            "drain",
            "pump",
            "lat_kast",
            "lai_noevap",
            "sw_init",
            "surq_lag",
            "msk_co1",
            "msk_co2",
            "msk_x",
            "evap_adj",
            "scoef",
            "surq_exp",
        ],
        "bounds": [
            [0.51, 179.0],
            [0.01, 99.9],
            [0.01, 0.99],
            [0.01, 0.99],
            [0.01, 0.99],
            [0.01, 0.99],
            [0.71, 1.29],
            [0.01, 0.99],
            [30.0, 70.0],
            [50.0, 80.0],
            [70.0, 90.0],
            [80.0, 95.0],
            [0.0, 2.0],
            [5.0, 50.0],
            [2.0, 20.0],
            [10.0, 200.0],
            [0.11, 1.99],
            [0.01, 0.99],
            [0.021, 0.199],
            [0.001, 0.100],
            [0.01, 0.49],
            [0.1, 49.0],
            [0.1, 49.0],
            [-4.9, 4.9],
            [-4.9, 4.9],
            [0.1, 9.9],
            [0.01, 0.99],
            [0.1, 499.9],
            [0.01, 0.90],
            [0.1, 5999.9],
            [0.1, 99.9],
            [0.1, 99.9],
            [3.1, 39.9],
            [7601, 29999],
            [10.1, 50.9],
            [0.1, 9.9],
            [0.02, 3.99],
            [0.1, 9.9],
            [0.01, 0.99],
            [1.1, 23.9],
            [0.1, 9.9],
            [0.1, 9.9],
            [0.01, 0.29],
            [0.51, 0.99],
            [0.01, 0.99],
            [1.1, 2.9],
        ]
    }

    par = sample.sample(problem,
                        N=6,
                        num_levels=6,
                        seed=1,
                        )

    # 源文件路径和复制文件路径
    cwd = "E:/4_CodeLearn/Python/SPOTPY-and-pySWATPlus"
    proj_path = os.path.join(cwd, "SA_TxtInOut")
    copy_path = os.path.join(cwd, "SA_copy")

    # 设置SWAT模拟时间范围
    start_sim = "2017-01-01"
    end_sim = "2017-12-31"

    # 设置SWAT输出时间范围
    start_print = "2017-07-01"
    end_print = "2017-12-31"
    warmup = 0

    # 输出选项
    show_output = True
    delete_copy = False

    reader = TxtinoutReader(proj_path)
    reader.enable_object_in_print_prt("basin_aqu", False, True, False, False)
    reader.set_simulation_time(start_sim, end_sim)
    reader.set_print_time(start_print, end_print, warmup)

    num_runs = par.shape[0]
    params_list = []
    # 将模型运行一次的参数进行封装,存入一个变量如a
    # 创建列表，将很多类似a的变量放入，比如叫a_list
    # 注意深复制和浅复制
    for i in range(5):
        swat_params = {"hydrology.hyd": ("name", [(None, "lat_ttime", par[i, 0]),
                                                  (None, "can_max", par[i, 1]),
                                                  (None, "esco", par[i, 2]),
                                                  (None, "epco", par[i, 3]),
                                                  (None, "cn3_swf", par[i, 4]),
                                                  (None, "perco", par[i, 5]),
                                                  (None, "pet_co", par[i, 6]),
                                                  (None, "latq_co", par[i, 7]),
                                                  ],
                                         ),
                       "cntable.lum": ("description", [(None, "cn_a", par[i, 8]),
                                                       (None, "cn_b", par[i, 9]),
                                                       (None, "cn_c", par[i, 10]),
                                                       (None, "cn_d", par[i, 11]),
                                                       ],
                                       ),
                       "aquifer.aqu": ("name", [(None, "gw_flo", par[i, 12]),
                                                (None, "dep_bot", par[i, 13]),
                                                (None, "dep_wt", par[i, 14]),
                                                (None, "flo_dist", par[i, 15]),
                                                (None, "bf_max", par[i, 16]),
                                                (None, "alpha_bf", par[i, 17]),
                                                (None, "revap", par[i, 18]),
                                                (None, "rchg_dp", par[i, 19]),
                                                (None, "spec_yld", par[i, 20]),
                                                (None, "flo_min", par[i, 21]),
                                                (None, "revap_min", par[i, 22]),
                                                ],
                                       ),
                       "snow.sno": ("name", [(None, "fall_tmp", par[i, 23]),
                                             (None, "melt_tmp", par[i, 24]),
                                             (None, "melt_max", par[i, 25]),
                                             (None, "melt_min", par[i, 25]),
                                             (None, "tmp_lag", par[i, 26]),
                                             (None, "snow_h2o", par[i, 27]),
                                             (None, "cov50", par[i, 28]),
                                             ],
                                    ),
                       "tiledrain.str": ("name", [(None, "dp", par[i, 29]),
                                                  (None, "t_fc", par[i, 30]),
                                                  (None, "lag", par[i, 31]),
                                                  (None, "rad", par[i, 32]),
                                                  (None, "dist", par[i, 33]),
                                                  (None, "drain", par[i, 34]),
                                                  (None, "pump", par[i, 35]),
                                                  (None, "lat_ksat", par[i, 36]),
                                                  ],
                                         ),
                       "parameters.bsn": ("igen", [(None, "lai_noevap", par[i, 37]),
                                                   (None, "sw_init", par[i, 38]),
                                                   (None, "surq_lag", par[i, 39]),
                                                   (None, "msk_co1", par[i, 40]),
                                                   (None, "msk_co2", par[i, 41]),
                                                   (None, "msk_x", par[i, 42]),
                                                   (None, "evap_adj", par[i, 43]),
                                                   (None, "scoef", par[i, 44]),
                                                   (None, "surq_exp", par[i, 45]),
                                                   ],
                                          ),
                       }
        params_list.append((cwd, reader, swat_params, copy_path, show_output, delete_copy))

    print("===============55555555555555555=================")

    p = multiprocessing.Pool()
    # 第二个参数为alist
    result = p.map(parallel_swat, params_list)
    save_list_to_file(result, "SA.txt")
    Si = analyze.analyze(problem, par, result)
    print(Si["S1"])
    # with multiprocessing.Manager() as manager:
    #     res = manager.list()
    #     p1 = multiprocessing.Process(target=huron_swat,
    #                                  args=(res, reader, swat_par_list[0], copy_path, show_output, delete_copy))
    #     p2 = multiprocessing.Process(target=huron_swat,
    #                                  args=(res, reader, swat_par_list[1], copy_path, show_output, delete_copy))
    #     p1.start()
    #     p2.start()
    #
    #     p1.join()
    #     p2.join()
    #
    #     print(res)
