lum.dtl: written by SWAT+ editor v2.3.3 on 2024-01-30 23:01 for SWAT+ rev.60.5.7
41

name                     conds      alts      acts       !plant and harvest for continuous summer crop
pl_hv_summer1                6         5         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4      alt5  
phu_base0                  hru         0              null                 -       0.15000         >         -         -         -         -       !base0 heat units to trigger plant
phu_base0                  hru         0              null                 -       0.30000         -         -         -         -         >       !plant even if soil water is high
phu_plant                  hru         0           phu_mat                 -       1.15000         -         >         -         -         -       !plant hu to trigger harvest
soil_water                 hru         0                fc                 *       2.00000         <         <         -         -         -       !don't schedule if too wet
jday                       hru         0              null                 -     350.00000         -         -         =         -         -       !harvest day even if too wet
year_rot                   hru         0              null                 -       1.00000         -         -         -         >         -       !reset rotation year
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
plant                      hru         0             plant              crop       0.00000       1.00000              null  y   n   n   n   y   
harvest_kill               hru         0        grain_harv              crop       0.00000       1.00000             grain  n   y   y   n   n   
rot_reset                  hru         0           reset_1              null       1.00000       1.00000              null  n   n   n   y   n   

name                     conds      alts      acts       !plant and harvest for 2  year summer crop rotation
pl_hv_summer2                7         9         5  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4      alt5      alt6      alt7      alt8      alt9  
jday                       hru         0              null                 -     350.00000         -         -         -         -         =         =         -         -         -       !harvest day even if too wet
phu_base0                  hru         0              null                 -       0.15000         >         >         -         -         -         -         -         -         -       !base0 heat units to trigger plant
phu_base0                  hru         0              null                 -       0.30000         -         -         -         -         -         -         -         >         >       !base0 heat units even if too wet
phu_plant                  hru         0           phu_mat                 -       1.15000         -         -         >         >         -         -         -         -         -       !plant hu to trigger harvest
soil_water                 hru         0                fc                 *       2.00000         <         <         <         <         -         -         -         -         -       !don't schedule if too wet
year_rot                   hru         0              null                 -       1.00000         =         -         =         -         =         -         -         =         -       !plant/harv crop 1 in rotation year 1
year_rot                   hru         0              null                 -       2.00000         -         =         -         =         -         =         >         -         =       !plant/harv crop 2 in rotation year 1 and reset
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
plant                      hru         0       plant_crop1             crop1       0.00000       0.00000              null  y   n   n   n   n   n   n   y   n   
plant                      hru         0       plant_crop2             crop2       0.00000       0.00000              null  n   y   n   n   n   n   n   n   y   
harvest_kill               hru         0        grain_harv             crop1       0.00000       0.00000              null  n   n   y   n   y   n   n   n   n   
harvest_kill               hru         0        grain_harv             crop2       0.00000       0.00000              null  n   n   n   y   n   y   n   n   n   
rot_reset                  hru         0           reset_2              null       1.00000       1.00000              null  n   n   n   n   n   n   y   n   n   

name                     conds      alts      acts       !plant and harvest for continuous winter crop
pl_hv_winter1                6         4         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4  
jday                       hru         0              null                 -     200.00000         -         -         =         -       !harvest day even if too wet
phu_base0                  hru         0              null                 -       0.85000         >         -         -         -       !base0 heat units to trigger plant
phu_plant                  hru         0           phu_mat                 -       1.15000         -         >         -         -       !plant heat units to trigger harvest
soil_water                 hru         0                fc                 *       2.00000         <         <         -         -       !don't schedule if too wet
year_rot                   hru         0              null                 -       1.00000         -         -         -         >       !reset rotation year
year_rot                   hru         0              null                 -       1.00000         =         =         =         -       !plant/harv crop 1 in rotation year 1
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
harvest_kill               hru         0        grain_harv              crop       0.00000       0.00000              null  n   y   y   n   
plant                      hru         0             plant              crop       0.00000       0.00000              null  y   n   n   n   
rot_reset                  hru         0           reset_1              null       1.00000       1.00000              null  n   n   n   y   

name                     conds      alts      acts       !corn-corn-soybean-winter wheat-soybean rotation
pl_hv_ccsws                 10        10         7  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4      alt5      alt6      alt7      alt8      alt9     alt10  
days_harv                  hru         0              null                 -       7.00000         -         -         =         =         -         -         -         -         -         -       !plant wwht and soyb 7 days after harvest
jday                       hru         0              null                 -     150.00000         -         -         -         -         -         -         -         -         -         =       !harv wwht on day 150 even if too wet
jday                       hru         0              null                 -     350.00000         -         -         -         -         -         -         -         =         =         -       !harv corn and soyb on day 350 even if too wet
phu_base0                  hru         0              null                 -       0.15000         >         >         -         -         -         -         -         -         -         -       !base0 heat units to plant corn and soyb
phu_plant                  hru         0           phu_mat                 -       1.15000         -         -         -         -         >         >         >         -         -         -       !plant heat units to harv corn and soyb
soil_water                 hru         0                fc                 *       2.00000         <         <         -         <         <         <         <         -         -         -       !don't schedule if too wet
year_rot                   hru         0              null                 -       4.00000         -         -         =         -         -         -         =         -         -         =       !rotation year 4 - harv wwht and plant/harv soyb
year_rot                   hru         0              null                 -       3.00000         -         =         -         =         -         -         -         -         -         -       !rotation year 3 - plant/harv soyb - plant wwht
year_rot                   hru         0              null                 -       2.00000         -         -         -         -         -         >         -         -         >         -       !rotation year > 2 (3 and 4) - harv soyb
year_rot                   hru         0              null                 -       3.00000         <         -         -         -         <         -         -         <         -         -       !rotation year < 3 (1 and 2) - plant/harv corn
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
harvest_kill               hru         0        grain_harv              wwht       0.00000       0.00000              null  n   n   n   n   n   n   y   n   n   y   
harvest_kill               hru         0        grain_harv              soyb       0.00000       0.00000              null  n   n   n   n   n   y   n   n   y   n   
harvest_kill               hru         0        grain_harv              corn       0.00000       0.00000              null  n   n   n   n   y   n   n   y   n   n   
plant                      hru         0        plant_wwht              wwht       0.00000       0.00000              null  n   n   n   y   n   n   n   n   n   n   
plant                      hru         0        plant_soyb              soyb       0.00000       0.00000              null  n   y   y   n   n   n   n   n   n   n   
plant                      hru         0        plant_corn              corn       0.00000       0.00000              null  y   n   n   n   n   n   n   n   n   n   
rot_reset                  hru         0           reset_4              null       1.00000       1.00000              null  n   n   n   n   n   n   n   n   n   n   

name                     conds      alts      acts       !irrigate 25 mm when soil water is below 0.5*field capacity from unlimited source - max 20 times per year using sprinkler
irr_opt_sw_unlim             2         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
soil_water                 hru         0                fc                 -       0.50000         <       !irrigate when soil water is below this fraction of field capacity
plant_gro                  hru         0                 y                 -       0.00000         =       !only when plant is growing
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irr_demand                 hru         0         sprinkler     sprinkler_ilm      25.00000      20.00000             unlim  y   

name                     conds      alts      acts       !irrigate 25 mm when soil water is below 0.35*field capacity from unlimited source - max 20 times per year using sprinkler
irr_reduc_sw_unlim           2         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
soil_water                 hru         0                fc                 -       0.35000         <       !irrigate when soil water is below this fraction of field capacity
plant_gro                  hru         0                 y                 -       0.00000         =       !only when plant is growing
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irr_demand                 hru         0         sprinkler     sprinkler_ilm      25.00000      20.00000             unlim  y   

name                     conds      alts      acts       !irrigate 25 mm when plant water stress < 0.9 from unlimited source - max 20 times per year using sprinkler
irr_str9_unlim               1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.90000         <       !irrigate when plant water stress is less than this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irr_demand                 hru         0         sprinkler     sprinkler_ilm      25.00000      20.00000             unlim  y   

name                     conds      alts      acts       !irrigate 25 mm when plant water stress < 0.8 from unlimited source - max 20 times per year using sprinkler
irr_str8_unlim               1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.80000         <       !irrigate when plant water stress is less than this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irr_demand                 hru         0         sprinkler     sprinkler_ilm      25.00000      20.00000             unlim  y   

name                     conds      alts      acts       !irrigate 25 mm when plant water stress < 0.8 from aquifer 1 - max 20 times per year using sprinkler
irr_str8_aqu                 1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.80000         <       !irrigate when plant water stress is less than this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   aqu         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   

name                     conds      alts      acts       !irrigate 25 mm when plant water stress < 0.8 from reservoir 1 - max 20 times per year using sprinkler
irr_str8_res                 1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.80000         <       !irrigate when plant water stress is less than this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   res         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   

name                     conds      alts      acts       !irrigate 25 mm when plant water stress < 0.8 from channel 1 - max 20 times per year using sprinkler
irr_str8_cha                 1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.80000         <       !irrigate when plant water stress is less than this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   cha         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   

name                     conds      alts      acts       !when there is plant water stress, irrigate from a reservoir with an aquifer backup
irr_str8_res_aqu             3         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
w_stress                   hru         0              null                 -       0.80000         <         <       !water stress < this value
vol                        res         1              pvol                 *       0.75000         >         <       !reservoir volume > this fraction of principle volume
aqu_dep                    aqu         1              null                 -      50.00000         -         <       !aquifer depth < this many meters below surface
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   res         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   n   
irrigate                   aqu         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   y   

name                     conds      alts      acts       !open drains in spring and close after harvest - close in summer if low soil water and plant hu > 0.5
control_drainage             6         4         4  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4  
jday                       hru         0              null                 -      65.00000         =         -         -         -       !open drains on day 65
days_harv                  hru         0              null                 -      14.00000         -         >         -         -       !close drains 14 days after harvest
jday                       hru         0              null                 -     295.00000         -         >         -         -       !close drains on day 295
phu_plant                  hru         0              null                 -       0.50000         -         -         >         -       !plant heaat units > 0.5 - check for summer closing
soil_water                 hru         0                fc                 *       0.50000         -         -         <         -       !sw < 0.5*fc - check for summer closing
phu_plant                  hru         0              null                 -       1.00000         -         -         -         >       !open when plant heat units > 1.0
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
drain_control              hru         0       open_spring              null       1.00000       1.00000              null  y   n   n   n   
drain_control              hru         0        close_fall              null       0.00000       1.00000              null  n   y   n   n   
drain_control              hru         0      close_summer              null       0.00000       1.00000              null  n   n   y   n   
drain_control              hru         0       open_summer              null       1.00000       1.00000              null  n   n   n   y   

name                     conds      alts      acts       !hay cutting (biomass harv) when plant heat units > 0.5 - soil is below 2*fc - and biomass is > 3000 kg/ha - max 3 cuttings per year
hay_cutting                  3         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
soil_water                 hru         0                fc                 *       2.00000         <       !soil water < this value * field capacity
phu_plant                  hru         0           phu_mat                 -       0.50000         >       !plant heat units > this value
biomass                    hru         0              null                 -    3000.00000         >       !biomass > this value (kg/ha)
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
harvest                    hru         0          hay_harv              alfa       0.00000       3.00000           biomass  y   

name                     conds      alts      acts       !forest cutting after 20 years of growth on day 295
forest_cut                   2         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
cur_yrs_mat                hru         0              null                 -      20.00000         =       !years of growth
jday                       hru         0              null                 -     295.00000         =       !day of year
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
harvest                    hru         0         tree_harv              frsd       0.00000       0.00000        forest_cut  y   

name                     conds      alts      acts       !graze after day 295 until day 85 when biomass > 3000 kg/ha - low beef cattle density from graze.ops
graze_winter                 3         2         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
jday                       hru         0              null                 -     295.00000         >         -       !day of year to start grazing
jday                       hru         0              null                 -      85.00000         -         <       !day of year to end grazing
biomass                    hru         0              null                 -    3000.00000         >         >       !remove animals if biomass (kg/ha) falls below this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
graze                      hru         0        winter_low          beef_low       0.00000       0.00000              null  y   y   

name                     conds      alts      acts       !graze after day 120 until day 200 when biomass > 3000 kg/ha - high beef cattle density from graze.ops
graze_summer                 3         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
jday                       hru         0              null                 -     120.00000         >       !day of year to start grazing
jday                       hru         0              null                 -     200.00000         <       !day of year to end grazing
biomass                    hru         0              null                 -    3000.00000         >       !remove animals if biomass (kg/ha) falls below this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
graze                      hru         0       summer_high         beef_high       0.00000       0.00000              null  y   

name                     conds      alts      acts       !160 kg/ha anh- ammonia in spring with max 2 50 kg/ha urea side dressings; 30 kg/ha phosphorus (once) when stress < 0.9
fert_sprg_side               5         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hru         0              null                 -       0.12000         >         -         -       !base0 heat units > this value (just before planting)
year_rot                   hru         0              null                 -       1.00000         =         =         -       !rotation year
p_stress                   hru         0              null                 -       0.90000         -         -         <       !phosphorus stress < this value
plant_gro                  hru         0                 y                 -       0.00000         -         =         =       !plant is growing
phu_plant                  hru         0           phu_mat                 -       1.05000         -         <         <       !heat units to maturity - don't fertilize after plant reaches maturity
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
fertilize                  hru         0    ammonia_spring           anh_nh3     160.00000       1.00000            inject  y   n   n   
fertilize                  hru         0        side_dress              urea      50.00000       2.00000         broadcast  n   y   n   
fertilize                  hru         0        side_dress            elem_p      30.00000       1.00000         broadcast  n   n   y   

name                     conds      alts      acts       !fertilizer stress test
fert_stess_test              4         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
n_stress                   hru         0              null                 -       0.90000         <         -       !nitrogen stress < this value
p_stress                   hru         0              null                 -       0.90000         -         <       !phosphorus stress < this value
plant_gro                  hru         0                 y                 -       1.00000         =         =       !plant is growing
phu_plant                  hru         0           phu_mat                 -       1.05000         <         <       !heat units to maturity - don't fertilize after plant reaches maturity
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
#change_params fertilize                  hru         0        side_dress            elem_n      #fert_amount#       5.00000         broadcast  y   n   
#change_params fertilize                  hru         0        side_dress            elem_p      #fert_amount#       2.00000         broadcast  n   y   

name                     conds      alts      acts       !inject 0.33 kg/ha of pesticide ffa randomly (unformly distributed) to 30% (0.3) of fields between day 121 and 181 - 1 time per year
pst_unif                     1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
prob_unif                  0.3       121              null                 -     181.00000         =       !random uniform distr of 30% of hru between days 121-181
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
pest_apply                 hru         0           ffa_app               ffa       0.33000       1.00000            inject  y   

name                     conds      alts      acts       !fall plow followed by field cultivator with spring field cultivator
fall_plow                    3         3         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hru         0              null                 -       0.12000         -         -         >       !field cutivator just before planting if base0 heat units > this value
days_harv                  hru         0              null                 -       3.00000         =         -         -       !days to plow after harvest
days_harv                  hru         0              null                 -       6.00000         -         =         -       !follow with field cultivator this many days after harvest
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
tillage                    hru         0     moldboardplow          mldboard       0.00000       1.00000              null  y   n   n   
tillage                    hru         0     fieldcultivat           fldcult       0.00000       2.00000              null  n   y   y   

name                     conds      alts      acts       !spring plow followed by field cultivator with fall field cultivator
spring_plow                  3         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hru         0              null                 -       0.11000         >         -         -       !plow just before planting if base0 heat units > this value
phu_base0                  hru         0              null                 -       0.12000         -         >         -       !follow with field cutivator if base0 heat units > this value
days_harv                  hru         0              null                 -       3.00000         -         -         =       !days after harvest to chisel plow
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
tillage                    hru         0     moldboardplow          mldboard       0.00000       1.00000              null  y   n   n   
tillage                    hru         0     fieldcultivat           fldcult       0.00000       1.00000              null  n   y   n   
tillage                    hru         0       chisel_plow          chisplow       0.00000       1.00000              null  n   n   y   

name                     conds      alts      acts       !mulch till - disk before plant and chisel plow after harvest
mulch_till_1                 2         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hru         0              null                 -       0.12000         >         -       ! disk just before planting if base0 heat units > this value
days_harv                  hru         0              null                 -       3.00000         -         =       !days after harvest to chisel plow
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
tillage                    hru         0       tandem_disk          tandemrg       0.00000       1.00000              null  y   n   
tillage                    hru         0       chisel_plow          chisplow       0.00000       1.00000              null  n   y   

name                     conds      alts      acts       !mulch till - chisel plow and disk before planting - no fall till
mulch_till_2                 2         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hru         0              null                 -       0.11000         >         -       !chisel plow just before planting if base0 heat units > this value
phu_base0                  hru         0              null                 -       0.12000         -         >       !follow with field cutivator if base0 heat units > this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
tillage                    hru         0       chisel_plow          chisplow       0.00000       1.00000              null  y   n   
tillage                    hru         0       tandem_disk          tandemrg       0.00000       1.00000              null  n   y   

name                     conds      alts      acts       !use zerotill in tillage.til to simulate a planter
no_till                      1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
phu_base0                  hru         0              null                 -       0.15000         >       !planter implement if base0 heat units > this value
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
tillage                    hru         0          planting          zerotill       0.00000       1.00000              null  y   

name                     conds      alts      acts  
cn2_update                   2         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_plant                  hru         0           phu_mat                 -       0.10000         >         -  
days_harv                  hru         0              null                 -       1.00000         -         =  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
cn_update                  hru         0     lower_cn_grow            abschg      -3.00000       1.00000              null  y   n   
cn_update                  hru         0  increase_cn_harv            abschg       6.00000       1.00000              null  n   y   

name                     conds      alts      acts  
irrig_rel                    3         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
vol                        res         0              evol                 *       1.00000         <         >         -  
vol                        res         0              pvol                 *       1.00000         >         -         -  
irr_dmd_wro                wro         1              null                 -       0.00000         -         -         >  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
release                    res         0    over_emergency              days       5.00000       0.00000              evol  n   y   n   
release                    res         0    over_principal              days      15.00000       0.00000              pvol  y   n   n   
allocate_wro               wro         1      direct_irrig               res       1.00000       0.00000              null  n   n   y   

name                     conds      alts      acts  
irr_str8_dmd                 1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
w_stress                   hru         0              null                 -       0.80000         <  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irr_demand                 hru         0         sprinkler     sprinkler_med      25.00000      20.00000              null  y   

name                     conds      alts      acts       !plant and harvest for continuous summer corn
pl_hv_summer1_corn           6         5         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4      alt5  
phu_base0                  hru         0              null                 -       0.15000         >         -         -         -         -       !base0 heat units to trigger plant
phu_base0                  hru         0              null                 -       0.30000         -         -         -         -         >       !plant even if soil water is high
phu_plant                  hru         0           phu_mat                 -       1.15000         -         >         -         -         -       !plant hu to trigger harvest
soil_water                 hru         0                fc                 *       2.00000         <         <         -         -         -       !don't schedule if too wet
jday                       hru         0              null                 -     350.00000         -         -         =         -         -       !harvest day even if too wet
year_rot                   hru         0              null                 -       1.00000         -         -         -         >         -       !reset rotation year
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
plant                      hru         0        plant_corn              corn       0.00000       1.00000              null  y   n   n   n   y   
harvest_kill               hru         0        grain_harv              corn       0.00000       1.00000             grain  n   y   y   n   n   
rot_reset                  hru         0           reset_1              null       1.00000       1.00000              null  n   n   n   y   n   

name                     conds      alts      acts       !plant and harvest for 2  year summer crop rotation - corn and soybean
pl_hv_summer2_corn_soyb         7         9         5  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3      alt4      alt5      alt6      alt7      alt8      alt9  
jday                       hru         0              null                 -     350.00000         -         -         -         -         =         =         -         -         -       !harvest day even if too wet
phu_base0                  hru         0              null                 -       0.15000         >         >         -         -         -         -         -         -         -       !base0 heat units to trigger plant
phu_base0                  hru         0              null                 -       0.30000         -         -         -         -         -         -         -         >         >       !base0 heat units even if too wet
phu_plant                  hru         0           phu_mat                 -       1.15000         -         -         >         >         -         -         -         -         -       !plant hu to trigger harvest
soil_water                 hru         0                fc                 *       2.00000         <         <         <         <         -         -         -         -         -       !don't schedule if too wet
year_rot                   hru         0              null                 -       1.00000         =         -         =         -         =         -         -         =         -       !plant/harv crop 1 in rotation year 1
year_rot                   hru         0              null                 -       2.00000         -         =         -         =         -         =         >         -         =       !plant/harv crop 2 in rotation year 1 and reset
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
plant                      hru         0        plant_corn              corn       0.00000       0.00000              null  y   n   n   n   n   n   n   y   n   
plant                      hru         0        plant_soyb              soyb       0.00000       0.00000              null  n   y   n   n   n   n   n   n   y   
harvest_kill               hru         0        grain_harv              corn       0.00000       0.00000              null  n   n   y   n   y   n   n   n   n   
harvest_kill               hru         0        grain_harv              soyb       0.00000       0.00000              null  n   n   n   y   n   y   n   n   n   
rot_reset                  hru         0           reset_2              null       1.00000       1.00000              null  n   n   n   n   n   n   y   n   n   

name                     conds      alts      acts  
EnlistDuo_app                3         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hru         0              null                 -       0.10000         >         -  
prob                      null         0              null                 -       0.07000         <         -  
phu_base0                  hru         0              null                 -       0.15000         -         >  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
pest_apply                 hru         0          preplant          dacamine       1.00000       1.00000         broadcast  y   y   
pest_apply                 hru         0          preplant           roundup       2.00000       1.00000         broadcast  y   y   

name                     conds      alts      acts  
no_tile_drain                1         1         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1  
day_start                  hru         0              null                 -       1.00000         =  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
drain_control              hru         0      close_always              null       0.00000       1.00000              null  y   

name                     conds      alts      acts  
pl_grow_sum                  2         2         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hlt         0              null                 -       0.15000         >         <  
phu_base0                  hlt         0              null                 -       0.20000         <         -  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
grow_init                  hlt         0      start_growth              file       0.00000       0.00000              null  y   n   

name                     conds      alts      acts  
pl_end_sum                   2         3         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hlt         0              null                 -       0.50000         >         <         -  
jday                       hlt         0              null                 -     330.00000         -         -         =  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
grow_end                   hlt         0        end_growth              file       0.00000       0.00000              null  y   n   y   

name                     conds      alts      acts  
fert_rot_1                   5         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hru         0              null                 -       0.12000         >         -  
year_rot                   hru         0              null                 -       1.00000         =         =  
n_stress                   hru         0              null                 -       0.90000         -         <  
plant_gro                  hru         0                 y                 -       1.00000         -         =  
phu_plant                  hru         0           phu_mat                 -       1.05000         -         <  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
fertilize                  hru         0    ammonia_spring           anh_nh3     160.00000       1.00000            inject  y   n   
fertilize                  hru         0        side_dress              urea      50.00000       2.00000         broadcast  n   y   

name                     conds      alts      acts  
pl_grow_win                  2         3         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hlt         0              null                 -       0.75000         >         <         -  
jday                       hlt         0              null                 -     330.00000         -         -         =  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
grow_init                  hlt         0      start_growth              wwht       0.00000       0.00000              null  y   n   y   

name                     conds      alts      acts  
pl_end_win                   2         2         1  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
phu_base0                  hlt         0              null                 -       0.20000         >         <  
phu_base0                  hlt         0              null                 -       0.25000         <         -  
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
grow_end                   hlt         0        end_growth             grain       0.00000       0.00000              null  y   n   

name                     conds      alts      acts       !when there is plant water stress, irrigate from a reservoir with an aquifer backup (and unlimited backup to both)
irr_str8_r_a_u               3         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
w_stress                   hru         0              null                 -       0.80000         <         <         <       !water stress < this value
vol                        res         1              pvol                 *       0.75000         >         <         <       !reservoir volume > this fraction of principle volume
aqu_dep                    aqu         1              null                 -      50.00000         -         <         >       !aquifer depth < this many meters below surface
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   res         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   n   n   
irrigate                   aqu         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   y   n   
irrigate                 unlim         0         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   n   y   

name                     conds      alts      acts       !when there is plant water stress, irrigate from an aquifer with a reservoir backup (and unlimited backup to both)
irr_str8_a_r_u               3         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
w_stress                   hru         0              null                 -       0.80000         <         <         <       !water stress < this value
aqu_dep                    aqu         1              null                 -      50.00000         <         >         >       !aquifer depth < this many meters below surface
vol                        res         1              pvol                 *       0.75000         -         >         <       !reservoir volume > this fraction of principle volume
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   aqu         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   n   n   
irrigate                   res         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   y   n   
irrigate                 unlim         0         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   n   y   

name                     conds      alts      acts       !when there is plant water stress, irrigate from an aquifer with a reservoir backup
irr_str8_aqu_res             3         2         2  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2  
w_stress                   hru         0              null                 -       0.80000         <         <       !water stress < this value
aqu_dep                    aqu         1              null                 -      50.00000         <         >       !aquifer depth < this many meters below surface
vol                        res         1              pvol                 *       0.75000         -         >       !reservoir volume > this fraction of principle volume
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
irrigate                   aqu         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  y   n   
irrigate                   res         1         sprinkler     sprinkler_ilm      25.00000      20.00000              null  n   y   

name                     conds      alts      acts       !spring side dressing
fert_sprg_side2              5         3         3  
var                        obj   obj_num           lim_var            lim_op     lim_const      alt1      alt2      alt3  
phu_base0                  hru         0              null                 -       0.12000         >         -         -       !base0 heat units > this value (just before planting)
year_rot                   hru         0              null                 -       1.00000         =         =         -       !rotation year
p_stress                   hru         0              null                 -       0.90000         -         -         <       !phosphorus stress < this value
plant_gro                  hru         0                 y                 -       0.00000         -         =         =       !plant is growing
phu_plant                  hru         0           phu_mat                 -       1.05000         -         <         <       !heat units to maturity - don't fertilize after plant reaches maturity
act_typ                    obj   obj_num              name            option         const        const2                fp  outcome           
fertilize                  hru         0    ammonia_spring           anh_nh3     160.00000       1.00000            inject  y   n   n   
fertilize                  hru         0        side_dress              urea      50.00000       2.00000         broadcast  n   y   n   
fertilize                  hru         0        side_dress            elem_p      30.00000       1.00000         broadcast  n   n   y   
