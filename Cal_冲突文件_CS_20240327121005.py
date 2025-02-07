#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""
一、TxtinoutReader
1.必须指定工作路径，否则工作路径会变成copy里的路径
2.必须使用绝对路径
3.网页教程的参数格式不对
4.改变参数，不指定行id时，用 None

二、FileReader
1.读取文件不指定index时，必须用None,不可省略
2.usecols省略的话，读取的df为空
"""
# %reset -f
import os
import pandas as pd
import numpy as np
import spotpy as sp
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as gauss
import shutil
import mpi4py
import sys
from pySWATPlus.TxtinoutReader import TxtinoutReader
from pySWATPlus.FileReader import FileReader
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm

# cwd = "E:/SPOTPY-and-pySWATPlus"
cwd = "E:/BaiduSyncdisk/Code/Python/SPOTPY-and-pySWATPlus"

# # 用pySWATPlus定义huron_swat函数

# In[10]:


def huron_swat(reader, params, tpl_params, copy_path, output_scale="day", show_output=False, delete_copy=True):
    result = reader.copy_and_run(dir=copy_path,
                                 params=params,
                                 tpl_params=tpl_params,
                                 show_output=show_output
                                 )
    reader = FileReader(os.path.join(result, "channel_sd_day.txt"),
                        has_units=True,
                        index=None,
                        usecols=["mon", "day", "yr", "unit", "flo_out"],
                        filter_by={"unit": 1}
                        )
    res = reader.df
    if output_scale == "mon":
        res = (res.
               groupby(["yr", "mon"]).
               agg({"flo_out": "mean"}).
               reset_index())
        res["Date"] = pd.to_datetime(pd.DataFrame({"year": res["yr"],
                                                   "month": res["mon"],
                                                   "day": 1}))
        res.drop(columns=["yr", "mon"], inplace=True)
    elif output_scale == "day":
        res["Date"] = pd.to_datetime(pd.DataFrame({"year": res["yr"],
                                                   "month": res["mon"],
                                                   "day": res["day"]}))
        res.drop(columns=["mon", "day", "yr", "unit"], inplace=True)
    if delete_copy:
        shutil.rmtree(result, ignore_errors=True)
    os.chdir(cwd) #改回当前路径
    return res


# # 定义SPOTPY类

# In[11]:

class spot_swat():
    def __init__(self, TxtInOut_abspath, copy_path, start_print, end_print,
                 output_scale="day", prior=sp.parameter.Uniform, obj_func=None,
                 show_output=False, delete_copy=True):
        self.reader = TxtinoutReader(TxtInOut_abspath)
        self.copy_path = copy_path
        self.start = start_print
        self.end = end_print
        self.output_scale = output_scale
        self.obj_func = obj_func
        self.show_output = show_output
        self.delete_copy = delete_copy

        self.params = [prior('alpha_bf', 0.0001, 0.9999), #0
                        prior('bf_max', 0.0001, 1.9999),
                        prior('dep_bot', 0.0001, 9.9999),
                        prior('dep_wt', 0.0001, 9.9999),
                        prior('flo_dist', 0.0001, 199.9999),
                        prior('flo_min', 0.0001, 49.9999),
                        prior('gw_flo', 0.0001, 1.9999),
                        # prior('no3_n', 0.0001, 999.9999),
                        prior('rchg_dp', 0.0001, 0.9999),
                        prior('revap', 0.0201, 0.1999), #9
                        prior('revap_min', 0.0001, 49.9999),
                        prior('spec_yld', 0.0001, 0.4999),
                        # prior('hl_no3n', 0.0001, 199.9999),
                        prior('cn_a', 30.0001, 69.9999),
                        prior('cn_b', 50.0001, 79.9999),
                        prior('cn_c', 70.0001, 89.9999),
                        prior('cn_d', 80.0001, 94.9999),
                        prior('can_max', 0.0001, 99.9999),
                        prior('cn3_swf', 0.0001, 0.9999),
                        prior('epco', 0.0001, 0.9999), #19
                        prior('esco', 0.0001, 0.9999),
                        prior('lat_ttime', 0.5001, 179.9999),
                        prior('latq_co', 0.0001, 0.9999),
                        prior('perco', 0.0001, 0.9999),
                        prior('pet_co', 0.7001, 1.2999),
                        # prior('exp_co', 0.0001, 0.9999),
                        # prior('fr_hum_act', 0.0001, 0.9999),
                        # prior('hum_c_n', 8.0001, 11.9999),
                        # prior('nitrate', 0.0001, 99.9999),
                        prior('ovn', 0.0101, 0.6999), #29
                        # prior('denit_exp', 0.0001, 2.9999),
                        # prior('denit_frac', 0.0001, 0.9999),
                        prior('evap_adj', 0.5001, 0.9999),
                        prior('lai_noevap', 0.0001, 9.9999),
                        prior('msk_co1', 0.0001, 9.9999),
                        prior('msk_co2', 0.0001, 9.9999),
                        prior('msk_x', 0.0001, 0.2999),
                        # prior('n_fix_max', 1.0001, 19.9999),
                        # prior('n_perc', 0.0001, 0.9999),
                        # prior('n_uptake', 0.0001, 99.9999), #39
                        # prior('nperco_lchtile', 0.0001, 0.9999),
                        # prior('orgn_min', 0.0011, 0.0029),
                        # prior('rsd_cover', 0.1001, 0.4999),
                        # prior('rsd_decay', 0.0001, 0.0499),
                        # prior('rsd_decomp', 0.0201, 0.0999),
                        prior('scoef', 0.0001, 0.9999),
                        prior('surq_exp', 1.0001, 2.9999),
                        prior('surq_lag', 1.0001, 23.9999),
                        prior('sw_init', 0.0001, 0.9999),
                        prior('cov50', 0.0001, 0.90), #49
                        prior('fall_tmp', -4.9999, 4.9999),
                        prior('melt_max_min', 0.0001, 9.9999),
                        prior('melt_tmp', -4.9999, 4.9999),
                        prior('snow_h2o', 0.0001, 499.9999),
                        prior('snow_init', 0.0001, 999.9999),
                        prior('tmp_lag', 0.0001, 0.9999),
                        prior('dist', 7600.0001, 29999.9999),
                        prior('dp', 0.0001, 5999.9999),
                        prior('drain', 10.0001, 50.9999),
                        prior('lag', 0.0001, 99.9999), #59
                        prior('lat_kast', 0.0101, 3.9999),
                        prior('pump', 0.0001, 9.9999),
                        prior('rad', 3.0001, 39.9999),
                        prior('t_fc', 0.0001, 99.9999),
                        # prior('fert_1', 0.0001, 999.9999),
                        # prior('fert_2', 0.0001, 999.9999),
                        # prior('fert_3', 0.0001, 999.9999),
                        # prior('fert_4', 0.0001, 999.9999),
                        # prior('fert_5', 0.0001, 999.9999),
                        # prior('fert_6', 0.0001, 999.9999),
                        # prior('fert_7', 0.0001, 999.9999),
                        # prior('fert_8', 0.0001, 999.9999),
                        # prior('fert_9', 0.0001, 999.9999),
                        # prior('fert_10', 0.0001, 999.9999),
                        # prior('fert_11', 0.0001, 999.9999),
                        # prior('fert_12', 0.0001, 999.9999),
                        # prior('rsd_init', 0.0001, 9999.9999),
                        prior('awc', 0.0001, 0.9999),
                        prior('soil_k', 0.0001, 1999.9999),
                        ]

        # self.params = [prior('alpha_bf', 0.0001, 0.9999), #0
        #                prior('dep_bot', 0.0001, 9.9999),
        #                prior('dep_wt', 0.0001, 9.9999),
        #                prior('flo_min', 0.0001, 49.9999),
        #                prior('gw_flo', 0.0001, 1.9999),
        #                prior('no3_n', 0.0001, 999.9999),
        #                prior('rchg_dp', 0.0001, 0.9999),
        #                prior('revap_min', 0.0001, 49.9999),
        #                prior('spec_yld', 0.0001, 0.4999),
        #                prior('hl_no3n', 0.0001, 199.9999), #9
        #                prior('esco', 0.0001, 0.9999),
        #                prior('latq_co', 0.0001, 0.9999),
        #                prior('perco', 0.0001, 0.9999),
        #                prior('cov50', 0.0001, 0.90),
        #                prior('fall_tmp', -4.9999, 4.9999),
        #                prior('melt_max_min', 0.0001, 9.9999),
        #                prior('melt_tmp', -4.9999, 4.9999),
        #                prior('snow_h2o', 0.0001, 499.9999),
        #                prior('snow_init', 0.0001, 999.9999),
        #                prior('dp', 0.0001, 5999.9999), # 19
        #                prior('fert_1', 0.0001, 999.9999),
        #                prior('fert_2', 0.0001, 999.9999),
        #                prior('fert_3', 0.0001, 999.9999),
        #                prior('fert_4', 0.0001, 999.9999),
        #                prior('fert_5', 0.0001, 999.9999),
        #                prior('fert_6', 0.0001, 999.9999),
        #                prior('fert_7', 0.0001, 999.9999),
        #                prior('fert_8', 0.0001, 999.9999),
        #                prior('fert_9', 0.0001, 999.9999),
        #                prior('fert_10', 0.0001, 999.9999), #29
        #                prior('fert_11', 0.0001, 999.9999),
        #                prior('fert_12', 0.0001, 999.9999),
        #                prior('awc', 0.0001, 0.9999),
        #                prior('soil_k', 0.0001, 1999.9999),
        #                ]
    def parameters(self):
        return sp.parameter.generate(self.params)

    def simulation(self, vector):
        par = np.array(vector)
        params = {"aquifer.aqu":("name", [(None, 'alpha_bf', par[0]),  
                                          (None, 'bf_max', par[1]),  
                                          (None, 'dep_bot', par[2]),  
                                          (None, 'dep_wt', par[3]),  
                                          (None, 'flo_dist', par[4]),  
                                          (None, 'flo_min', par[5]),  
                                          (None, 'gw_flo', par[6]),  
                                          # (None, 'no3_n', par[7]),  
                                          (None, 'rchg_dp', par[7]),  
                                          (None, 'revap', par[8]),  
                                          (None, 'revap_min', par[9]),  
                                          (None, 'spec_yld', par[10]),  
                                          # (None, 'hl_no3n', par[12]),  
                                          ],
                                  ),
                  "cntable.lum":("description", [(None,"cn_a", par[11]),
                                                  (None,"cn_b", par[12]), 
                                                  (None,"cn_c", par[13]), 
                                                  (None,"cn_d", par[14]), 
                                                  ],
                                  ),
                  "hydrology.hyd":("name", [(None, 'can_max', par[15]),
                                            (None, 'cn3_swf', par[16]),  
                                            (None, 'epco', par[17]),  
                                            (None, 'esco', par[18]),  
                                            (None, 'lat_ttime', par[19]),  
                                            (None, 'latq_co', par[20]),  
                                            (None, 'perco', par[21]),  
                                            (None, 'pet_co', par[22]),  
                                            ],
                                    ),
                  # "nutrients.sol":("name", [(None, 'exp_co', par[25]), 
                  #                           (None, 'fr_hum_act', par[26]),  
                  #                           (None, 'hum_c_n', par[27]),  
                  #                           (None, 'nitrate', par[28]), 
                  #                           ],
                  #                   ),
                  "ovn_table.lum":("name", [(None, 'ovn_mean', par[23]),  
                                            (None, 'ovn_min', par[23]),  
                                            (None, 'ovn_max', par[23]) 
                                            ],
                                    ),
                  "parameters.bsn":("igen", [
                                              # (None, 'denit_exp', par[30]),  
                                              # (None, 'denit_frac', par[31]), 
                                              (None, 'evap_adj', par[24]),  
                                              (None, 'lai_noevap', par[25]), 
                                              (None, 'msk_co1', par[26]), 
                                              (None, 'msk_co2', par[27]),  
                                              (None, 'msk_x', par[28]),  
                                              # (None, 'n_fix_max', par[37]),  
                                              # (None, 'n_perc', par[38]),  
                                              # (None, 'n_uptake', par[39]), 
                                              # (None, 'nperco_lchtile', par[40]),  
                                              # (None, 'orgn_min', par[41]),  
                                              # (None, 'rsd_cover', par[42]), 
                                              # (None, 'rsd_decay', par[43]),  
                                              # (None, 'rsd_decomp', par[44]),  
                                              (None, 'scoef', par[29]),  
                                              (None, 'surq_exp', par[30]), 
                                              (None, 'surq_lag', par[31]),  
                                              (None, 'sw_init', par[32]), 
                                              ],
                                    ),
                  "snow.sno":("name", [(None, 'cov50', par[33]),  
                                        (None, 'fall_tmp', par[34]),  
                                        (None, 'melt_max', par[35]),  
                                        (None, 'melt_min', par[35]),  
                                        (None, 'melt_tmp', par[36]),  
                                        (None, 'snow_h2o', par[37]), 
                                        (None, 'snow_init', par[38]), 
                                        (None, 'tmp_lag', par[38]), 
                                        ],
                              ),
                  "tiledrain.str":("name", [(None, 'dist', par[40]), 
                                            (None, 'dp', par[41]), 
                                            (None, 'drain', par[42]), 
                                            (None, 'lag', par[43]),  
                                            (None, 'lat_ksat', par[44]),  
                                            (None, 'pump', par[45]), 
                                            (None, 'rad', par[46]), 
                                            (None, 't_fc', par[47]), 
                                            ],
                                    ),
        
                  }
        tpl_params = {
            # "management.sch.tpl": {"fert_1": par[64],
            #                        "fert_2": par[65],
            #                        "fert_3": par[66],
            #                        "fert_4": par[67],
            #                        "fert_5": par[68],
            #                        "fert_6": par[69],
            #                        "fert_7": par[70],
            #                        "fert_8": par[71],
            #                        "fert_9": par[72],
            #                        "fert_10": par[73],
            #                        "fert_11": par[74],
            #                        "fert_12": par[75],
            #                        },
            # "plant.ini.tpl": {"rsd_init": par[76]}, 
            "soils.sol.tpl": {"awc": par[48],
                              "soil_k": par[49]},
                      }

        # params = {"aquifer.aqu":("name", [(None, 'alpha_bf', par[0]),  # 17
        #                                   (None, 'dep_bot', par[1]),  # 2
        #                                   (None, 'dep_wt', par[2]),  # 10
        #                                   (None, 'flo_min', par[3]),  # 3
        #                                   (None, 'gw_flo', par[4]),  # 22
        #                                   (None, 'no3_n', par[5]),  # 5
        #                                   (None, 'rchg_dp', par[6]),  # 23
        #                                   (None, 'revap_min', par[7]),  # 15
        #                                   (None, 'spec_yld', par[8]),  # 16
        #                                   (None, 'hl_no3n', par[9]),  # 1
        #                                   ],
        #                          ),
        #           "hydrology.hyd":("name", [
        #                                     (None, 'esco', par[10]),  # 20
        #                                     (None, 'latq_co', par[11]),  # 12
        #                                     (None, 'perco', par[12]),  # 11
        #                                     ],
        #                            ),
        #           "snow.sno":("name", [(None, 'cov50', par[13]),  # 6
        #                                (None, 'fall_tmp', par[14]),  # 19
        #                                (None, 'melt_max', par[15]),  # 13
        #                                (None, 'melt_min', par[15]),  # 14
        #                                (None, 'melt_tmp', par[16]),  # 7
        #                                (None, 'snow_h2o', par[17]),  # 18
        #                                (None, 'snow_init', par[18]),  # 21
        #                                ],
        #                       ),
        #           "tiledrain.str":("name", [
        #                                     (None, 'dp', par[19]),  # 8
        #                                     ],
        #                            ),

        #           }
        # tpl_params = {"management.sch.tpl": {"fert_1": par[20],
        #                                      "fert_2": par[21],
        #                                      "fert_3": par[22],
        #                                      "fert_4": par[23],
        #                                      "fert_5": par[24],
        #                                      "fert_6": par[25],
        #                                      "fert_7": par[26],
        #                                      "fert_8": par[27],
        #                                      "fert_9": par[28],
        #                                      "fert_10": par[29],
        #                                      "fert_11": par[30],
        #                                      "fert_12": par[31],
        #                                      },
        #               "soils.sol.tpl": {"awc": par[32],  # 9
        #                                 "soil_k": par[33]},  # 4
        #               }


        sim = huron_swat(self.reader, params, tpl_params, self.copy_path,
                         output_scale=self.output_scale,
                         show_output=self.show_output,
                         delete_copy=self.delete_copy)
        return sim["flo_out"]

    def evaluation(self):
        obs = pd.read_csv(os.path.join(cwd,'TimeSeries\\04199000.csv'))
        obs["Date"] = pd.to_datetime(obs["Date"])
        obs["year"] = obs["Date"].dt.year
        obs["mon"] = obs["Date"].dt.month
        if self.output_scale == "mon":
            obs = (obs.
                   groupby(["year", "mon"]).
                   agg({"Q":"mean"}).
                   reset_index()
                   )
            obs["Date"] = pd.to_datetime(pd.DataFrame({"year": obs["year"],
                                                       "month": obs["mon"],
                                                       "day": 1}))
            obs.drop(columns=["year", "mon"], inplace=True)
            
        obs = obs.loc[((obs["Date"] >= datetime.strptime(self.start, "%Y-%m-%d")) & (obs["Date"] <= datetime.strptime(self.end, "%Y-%m-%d"))),("Q", )]
        return obs
    def objectivefunction(self, simulation, evaluation):
        if not self.obj_func:
            like = sp.objectivefunctions.nashsutcliffe(evaluation, simulation)
        else:
            like = self.obj_func(evaluation, simulation)
        return like


# In[12]:
if __name__ == "__main__":
    # 源文件路径和复制文件路径
    proj_path = os.path.join(cwd, "Cal_TxtInOut")
    copy_path = os.path.join(cwd, "Cal_copy")

    # 设置SWAT模拟时间范围
    start_sim = "2017-01-01"
    end_sim = "2022-12-31"

    # 设置SWAT输出时间范围
    start_print = "2018-01-01"
    end_print = "2022-12-31"
    warmup = 1

    # 设置输出时间尺度
    output_scale = "mon"

    # 输出选项
    show_output = True
    delete_copy = False

    # 目标函数
    def obj_func(evaluation, simulation):
        evaluation = np.array(evaluation)
        simulation = np.array(simulation)
        e = np.where(simulation < evaluation[:, 0],
                     evaluation[:, 0] - simulation,
                     np.where(simulation > evaluation[:, 1],
                              evaluation[:, 1] - simulation,
                              0)
                     )
        # o = np.zeros_like(e)
        return np.sqrt(np.mean(e**2))

    # 实例化及采样
    spot_setup = spot_swat(proj_path, copy_path, start_print, end_print,
                           obj_func=None, output_scale=output_scale,
                           show_output=show_output, delete_copy=delete_copy)

    spot_setup.reader.set_simulation_time(start_sim, end_sim)
    spot_setup.reader.set_print_time(start_print, end_print, warmup)
    spot_setup.reader.enable_object_in_print_prt("channel_sd", True, False, False, False)

    sampler = sp.algorithms.lhs(spot_setup,
                                  dbname="Cal",
                                  dbformat="csv",
                                  # parallel="mpc",
                                  )
    # print(describe(sampler))
    rep = 1
    sampler.sample(rep,
                   # ngs,
                   )

    print("============= Successfully Well done! =================")
