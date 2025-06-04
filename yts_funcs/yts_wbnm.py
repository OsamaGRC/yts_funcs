import os, sys
import pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

def create_wbnm_storm(storm_df:str, coordinate_file:str, output_folder:str, timeshift:int, date_subset:list[str]=None, to_text:bool=True, return_output:bool=True, calc_step:float=1.0, out_step:float=15.0):
    '''
    storm_df:         the file containing rainfall data (expected to be in 15 min step)
    coordinate_file:    the file containing catchments coordinates [static file]
    output_folder:      the directory where results are to be outputed
    timeshift:          n*(15 min) ~ 44 for +11 hrs and 40 for +10 hrs
    date_subset:        enables clipping the data by [start_date, end_date]
    to_text:            store the output in disk if true
    return_output:      return the output of the function in the form of a variable
    '''

    if isinstance(storm_df, str):
        storm_df = Path(storm_df)
        df_rf = pd.read_csv(storm_df,parse_dates=[0], index_col="time")
    elif isinstance(storm_df, pd.DataFrame):
        df_rf = storm_df.copy()
    else:
        raise TypeError("storm_df must be a file path or a DataFrame.")
    
    coordinate_file = Path(coordinate_file)
    output_folder = Path(output_folder)

    df_co = pd.read_csv(coordinate_file)
    df_rf = df_rf[df_co["Name"]]
    
    if timeshift:
        df_rf= df_rf.shift(periods=timeshift, freq='15min')

    if date_subset:
        df_rf = df_rf.loc[date_subset[0]:date_subset[1]]
    
    start_dt = df_rf.index[0].strftime("%d%m%YT%H%M")
    end_dt = df_rf.index[-1].strftime("%d%m%YT%H%M")
    outputfile = f"{start_dt}+{int(timeshift*15/60)}TZ.txt"

    # Generating WBNM formmated storm
    block_head = ["#####START_STORM_BLOCK#############|###########|###########|###########|",
              "           1",
              "#####START_STORM#1",
              f"Python generated storm file at {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}",
              f"        {calc_step:.2f}", # Calculaion timestep 
              f"        {out_step:.2f}", # Output timestep
              "#####START_RECORDED_RAIN",
              f"{df_rf.index[0].strftime('%d/%m/%Y')}",
              f"{df_rf.index[0].strftime('%H:%M:%S')}",
              f"           {df_rf.shape[0]}       {'15'}",
              "MM/PERIOD",
              f"           {df_rf.shape[1]}"
              ]
    for i in range(len(df_rf.columns)):                                      
        # print(f"Creating storm for {df_rf.columns[i]}")
        block_head.append(f"Catchment: {df_rf.columns[i]}, No: {i+1}")
        block_head.append(f"    {df_co['x_coord'][i]}  {df_co['y_coord'][i]}")
        block_head.extend(df_rf.iloc[:,i].apply(str).to_list())
        
    block_head.append("#####END_RECORDED_RAIN")   
    block_head.append("#####START_CALC_RAINGAUGE_WEIGHTS")
    block_head.append("#####END_CALC_RAINGAUGE_WEIGHTS")

    if to_text:
        if not output_folder:
            output_folder = Path(os.getcwd())
        
        with open(Path.joinpath(output_folder,outputfile), 'w') as file:
            for item in block_head:
                file.write(item+"\n")
            print(f"Successfully created file: {Path.joinpath(output_folder,outputfile)}")
    if return_output:
        return block_head
    