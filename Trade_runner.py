from tradeoff_tool import trade_off, sens_anal
import pandas as pd
import os
import timeit

start = timeit.default_timer()

local_direc = os.path.abspath(__file__)
split_direc = os.path.split(local_direc)[0]
#TO_foldername = 'TO_Example'

#all weights should be between 0 and 1 in the file. Otherwise program will complain
criteria_weights_ex = pd.read_csv(os.path.join(split_direc,"Real_TO/CW_V1.csv"),float_precision='high')  

design_options_ex   = pd.read_csv(os.path.join(split_direc,"Real_TO/DO_V1.csv"),float_precision='high')

sens_anal_p_data    = pd.read_csv(os.path.join(split_direc,"Real_TO/Sens_Data.csv")) #Different std of p values for different mapping functions

to  = trade_off(criteria_weights_ex,design_options_ex)
a   = to.criteria






sens_anal(to,sens_anal_p_data)
c   = to.output_data
c.sort_values(by='Total Score',ascending=False)
print(c.to_markdown())
print(c.to_csv())
end = timeit.default_timer()
print('Sensetime:',end-start,'s')
#print(c.to_latex(index=False,))