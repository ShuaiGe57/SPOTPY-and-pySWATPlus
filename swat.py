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
import shutil
import mpi4py
import sys
from pySWATPlus.TxtinoutReader import TxtinoutReader
from pySWATPlus.FileReader import FileReader
from matplotlib import pyplot as plt
from datetime import datetime
cwd = os.getcwd()


# # 用pySWATPlus定义huron_swat函数

# In[10]:


def huron_swat(reader, params):
    result = reader.copy_and_run(dir=os.path.join(cwd,'try'),
                                 params=params,
                                 show_output=False
                                 )
    # print(reader.root_folder)
    reader = FileReader(os.path.join(result, "channel_sd_day.txt"),
                        has_units = True,
                        index=None,
                        usecols=["mon", "day", "yr", "unit", "flo_out"],
                        filter_by={"unit": 1}
                        )
    res = reader.df
    res["Date"] = pd.to_datetime(pd.DataFrame({"year": res["yr"],
                                                 "month": res["mon"],
                                                 "day": res["day"]}))
    res.drop(columns=["mon", "day", "yr", "unit"], inplace=True)
    os.chdir(cwd) #改回当前路径
    return res


# # 定义SPOTPY类

# In[11]:


from spotpy.parameter import Uniform
class spot_swat():
    def __init__(self, TxtInOut_abspath):
        self.reader = TxtinoutReader(TxtInOut_abspath)
        self.params = [Uniform("lat_ttime",0.51, 179.0),
                       Uniform("can_max",0.01, 99.9),
                       Uniform("esco",0.01,0.99),
                       Uniform("epco",0.01, 0.99),
                       Uniform("cn3_swf",0.01, 0.99),
                       Uniform("perco",0.01, 0.99),
                       Uniform("pet_co",0.71, 1.29),
                       Uniform("latq_co",0.01, 0.99),
                       Uniform("cn_a",50.0, 70.0),
                       Uniform("cn_b",70.0, 80.0),
                       Uniform("cn_c",80.0, 90.0),
                       Uniform("cn_d",90.0, 99.0),
                       ]
    def parameters(self):
        return sp.parameter.generate(self.params)
    # par = [Uniform("lat_ttime",0.5, 180.0),
    #        Uniform("can_max",0.0, 100.0),
    #        Uniform("esco",0.0,.0),
    #        Uniform("epco",0.0, 1.0),
    #        Uniform("cn3_swf",0.0, 1.0),
    #        Uniform("perco",0.0, 1.0),
    #        Uniform("pet_co",0.7, 1.3),
    #        Uniform("latq_co",0.0, 1.0),
    #        Uniform("cn_a",50.0, 70.0),
    #        Uniform("cn_b",70.0, 80.0),
    #        Uniform("cn_c",80.0, 90.0),
    #        Uniform("cn_d",90.0, 100.0),
    #        ]
    def simulation(self, vector):
        par = np.array(vector)
        swat_params = {"hydrology.hyd":("name", [(None,"lat_ttime", par[0]),
                                                 (None,"can_max", par[1]),
                                                 (None,"esco", par[2]),
                                                 (None,"epco", par[3]),
                                                 (None,"cn3_swf", par[4]),
                                                 (None,"perco", par[5]),
                                                 (None,"pet_co", par[6]),
                                                 (None,"latq_co", par[7]),
                                                 ],
                                        ),
                       "cntable.lum":("description", [("Row_crops","cn_a", par[8]),
                                                      ("Row_crops","cn_b", par[9]),
                                                      ("Row_crops","cn_c", par[10]),
                                                      ("Row_crops","cn_d", par[11]),
                                                      ],
                                      ),
                       }
        sim = huron_swat(self.reader ,swat_params)
        return sim["flo_out"]
    
    def evaluation(self):
        obs = pd.read_csv(os.path.join(cwd,'TimeSeries\\04199000.csv'))
        obs["Date"] = pd.to_datetime(obs["Date"])
        obs = obs.loc[((obs["Date"] >= "2018-01-01") & (obs["Date"] <= "2020-12-31")),"Q"]
        return obs
    def objectivefunction(self, simulation, evaluation):
        objfun = sp.objectivefunctions.nashsutcliffe(evaluation, simulation)
        return objfun
    


# In[12]:
proj_path = os.path.join(cwd, "TxtInOut")
print(proj_path)
spot_setup = spot_swat(proj_path)
sampler = sp.algorithms.sceua(spot_setup,
                                dbname="huron",
                                dbformat="csv",
                                parallel="mpi",
                                )
# print(describe(sampler))
sampler.sample(repetitions=2000,
               ngs=10,
               kstop=10,
               pcento=0.01,
               peps=0.01
               )
