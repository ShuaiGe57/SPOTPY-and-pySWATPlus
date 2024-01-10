# SPOTPY-and-pySWATPlus
利用SPOTPY和pySWATPlus进行SWAT模型率定、灵敏度分析、不确定性分析等研究
## 1.安装
  pySWATPlus 下载rar解压，用anaconda prompt界面cd到解压的文件夹，用pytohn setup.py install安装
  SPOTPY
  mpi4py
  MS mpi 下载.exe文件安装
  
## 2.注意
### 2.1 pySWATPlus
#### TxtinoutReader
1.必须指定工作路径，否则工作路径会变成copy里的路径
2.必须使用绝对路径
3.网页教程的参数格式不对
4.改变参数，不指定行id时，用 None

#### FileReader
1.读取文件不指定index时，必须用None,不可省略
2.usecols省略的话，读取的df为空

### 2.2 SPOTPY
#### 定义SWAT函数给SPOTPY用时注意以下几点
1、SWAT模型的参数定义不要超过上下界，如(0, 1)用(0.001, 0.999)代替
2、用pySWATPlus定义函数给simulation用时，尽量把只修改一次文件的语句放函数外面
   例如， reader.set_beginning_and_end_year(2017, 2018)
         reader.set_warmup(1)
         reader.enable_object_in_print_prt("channel_sd", True, False, False, False)
   这些直接在root文件夹里手动改就行，不要写成代码，因为用mpiexec并行处理时，会导致文件被意想不到的修改
