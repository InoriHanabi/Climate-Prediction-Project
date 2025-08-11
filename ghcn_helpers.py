# ghcn_helpers.py
import urllib.request
from datetime import date, datetime 
import numpy as np
import pandas as pd 
import os

class Station():
    def __init__(self,sid,lat,lon,el,state,name,gsn,hcn,wmo,country):
        self.sid=sid; self.lat=lat; self.lon=lon; self.el=el; self.state=state
        self.name=name; self.gsn=gsn; self.hcn=hcn; self.wmo=wmo; self.country=country
    def __str__(self):
        return f"{self.sid} is {self.name}, {self.country} at {self.lat}, {self.lon}, {self.el}"

class GHNCD:
    def __init__(self):
        self.station_col_len = [11,4,2,4] + [5,3]*31
    def chunkstring(self,string, lengths):
        return (string[pos:pos+length].strip() 
                for idx,length in enumerate(lengths) 
                for pos in [sum(lengths[:idx])])
    def processFile(self,fileName):
        outDict={}
        try:
            with open(fileName, 'r') as fp:
                for line in fp:
                    fields = list(self.chunkstring(line, self.station_col_len))
                    try:
                        year, month, field = int(fields[1]), int(fields[2]), fields[3]
                    except (ValueError, IndexError): continue
                    vals,flags = fields[4::2],fields[5::2]
                    def checkInt(x_str, mqs_flag_triplet_str): 
                        x_str = x_str.strip()
                        mqs_flag_triplet_str = mqs_flag_triplet_str.strip()
                        if x_str == '-9999': return -9999
                        q_flag = mqs_flag_triplet_str[1] if len(mqs_flag_triplet_str) >= 2 else ' '
                        if q_flag != ' ': return -9999
                        try: return int(x_str)
                        except ValueError: return -9999
                    ivals=[checkInt(x,f) for x,f in zip(vals,flags)]
                    monthDict={'year':year,'month':month,'field':field,'vals':ivals}
                    outDict.setdefault(field,{'monthList':[]})['monthList'].append(monthDict)
        except Exception: return {}
        return dict(outDict)
    def readCountriesFile(self,fileName=None, default_url='http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/ghcnd/ghcnd-countries.txt'):
        self.countryDict={}
        file_to_read = None
        try:
            if fileName is not None: file_to_read = open(fileName, 'r', encoding='utf-8')
            else: file_to_read = urllib.request.urlopen(default_url)
            for line_bytes in file_to_read:
                line = line_bytes.decode('utf-8') if isinstance(line_bytes, bytes) else line_bytes
                c=line[0:2].strip(); d=line[3:].strip()
                if c and d: self.countryDict[c]=d
        except Exception: pass
        finally:
            if file_to_read and hasattr(file_to_read, 'close'): file_to_read.close()
    def readStationsFile(self,fileName=None, justGSN=True, default_url='http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/ghcnd/ghcnd-stations.txt'):
        self.stationDict={}
        file_to_read = None
        if not hasattr(self, 'countryDict'): self.countryDict = {}
        try:
            if fileName is not None: file_to_read = open(fileName, 'r', encoding='utf-8')
            else: file_to_read = urllib.request.urlopen(default_url)
            for line_bytes in file_to_read:
                line = line_bytes.decode('utf-8') if isinstance(line_bytes, bytes) else line_bytes
                sid=line[0:11].strip()
                try:
                    lat=float(line[12:20].strip()); lon=float(line[21:30].strip()); el=float(line[31:37].strip())
                except ValueError: continue 
                state=line[38:40].strip(); name=line[41:71].strip(); gsn=line[72:75].strip()
                hcn=line[76:79].strip(); wmo=line[80:85].strip()
                if justGSN and not gsn: continue
                country_name = self.countryDict.get(sid[0:2], "Unknown")
                self.stationDict[sid]=Station(sid,lat,lon,el,state,name,gsn,hcn,wmo,country_name)
        except Exception: pass
        finally:
            if file_to_read and hasattr(file_to_read, 'close'): file_to_read.close()
    def getVar(self,statDict,varName='TMAX'):
        cal = 0.1
        if varName in ['SNOW','SNWD']: cal = 1.0
        if varName == 'PRCP': cal = 0.1
        tempList = []
        if varName not in statDict: return tempList
        for month_data in statDict[varName].get('monthList', []):
            year,month = month_data['year'],month_data['month']
            for ind, val in enumerate(month_data['vals']):
                if val != -9999:
                    try: tempList.append((date(year,month,ind+1), cal * val))
                    except ValueError: continue
        return tempList
    def getStation(self,sid): return self.stationDict.get(sid, None)
    def getStatKeyNames(self): return list(self.stationDict.keys())

class getdata:
    def __init__(self, ghn_instance, base_url='http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/ghcnd/ghcnd_gsn/'):
        self.ghn = ghn_instance; self.base_url = base_url
        self.statNames = self.ghn.getStatKeyNames()
    def _fetch_and_process_station(self, Whichstat_idx):
        if not self.statNames or not (0 <= Whichstat_idx < len(self.statNames)): return None, None
        station_id = self.statNames[Whichstat_idx]; fileName = station_id + '.dly'
        urlName = self.base_url + fileName
        try: urllib.request.urlretrieve(urlName, fileName)
        except Exception:
            if not os.path.exists(fileName): return None, None
        statDict = self.ghn.processFile(fileName)
        return (statDict, station_id) if statDict else (None, None)
    def TmaxTmin(self, Whichstat_idx):
        statDict, _ = self._fetch_and_process_station(Whichstat_idx)
        if not statDict: return [], [], [], [], Whichstat_idx
        tmaxA = self.ghn.getVar(statDict, 'TMAX'); dM, tM = zip(*tmaxA) if tmaxA else ([], [])
        tminA = self.ghn.getVar(statDict, 'TMIN'); dm, tm = zip(*tminA) if tminA else ([], [])
        return list(dM), list(tM), list(dm), list(tm), Whichstat_idx
    def PRCP(self,Whichstat_idx):
        statDict, _ = self._fetch_and_process_station(Whichstat_idx)
        if not statDict: return [],[],Whichstat_idx,"Rainfall Precipitate in millimeters"
        arr = self.ghn.getVar(statDict,'PRCP'); d,v = zip(*arr) if arr else ([],[])
        return list(d),list(v),Whichstat_idx,"Rainfall Precipitate in millimeters"
    def SNOW(self, Whichstat_idx):
        statDict, _ = self._fetch_and_process_station(Whichstat_idx)
        if not statDict: return [], [], Whichstat_idx, "Snowfall in millimeters"
        arr = self.ghn.getVar(statDict, 'SNOW'); days, snow = zip(*arr) if arr else ([], [])
        return list(days), list(snow), Whichstat_idx, "Snowfall in millimeters"
    def SNWD(self, Whichstat_idx):
        statDict, _ = self._fetch_and_process_station(Whichstat_idx)
        if not statDict: return [], [], Whichstat_idx, "Snow Depth in millimeters"
        arr = self.ghn.getVar(statDict, 'SNWD'); days, snwd = zip(*arr) if arr else ([], [])
        return list(days), list(snwd), Whichstat_idx, "Snow Depth in millimeters"

def _create_complete_df(days_raw, values_raw, col_name='value'):
    if not days_raw: return pd.DataFrame(columns=[col_name])
    valid_dates = [d for d in days_raw if isinstance(d, date)]
    if not valid_dates: return pd.DataFrame(columns=[col_name])
    df = pd.DataFrame({col_name: values_raw}, index=pd.to_datetime(valid_dates))
    if df.empty: return df
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    return df.reindex(full_date_range)

def fillholesT (Temp_data_tuple):
    days_max_raw, tmax_values_raw, days_min_raw, tmin_values_raw, _ = Temp_data_tuple
    dfmax = _create_complete_df(days_max_raw, tmax_values_raw, 'temperature')
    dfmin = _create_complete_df(days_min_raw, tmin_values_raw, 'temperature')
    if dfmax.empty and dfmin.empty: return np.array([]), np.array([]), np.array([]), np.array([]), "NC", "NC"
    for df in [dfmax, dfmin]:
        if df.empty: continue
        for (month, day), group in df.groupby([df.index.month, df.index.day]):
            missing_indices = group.index[group['temperature'].isna()]
            if not missing_indices.empty:
                avg_temp = df[(df.index.month == month) & (df.index.day == day) & (df['temperature'].notna())]['temperature'].mean()
                if pd.isna(avg_temp): avg_temp = df[(df.index.month == month) & (df['temperature'].notna())]['temperature'].mean()
                if pd.isna(avg_temp): avg_temp = df[df['temperature'].notna()]['temperature'].mean()
                if pd.isna(avg_temp): avg_temp = 0.0
                df.loc[missing_indices, 'temperature'] = np.random.uniform(avg_temp - 5, avg_temp + 5, size=len(missing_indices))
    confirmax = "C" if not dfmax.empty and dfmax.index[-1].year >= 2020 else "NC"
    confirmin = "C" if not dfmin.empty and dfmin.index[-1].year >= 2020 else "NC"
    return (dfmax.index.to_numpy() if not dfmax.empty else np.array([]),
            dfmax['temperature'].values if not dfmax.empty else np.array([]),
            dfmin.index.to_numpy() if not dfmin.empty else np.array([]),
            dfmin['temperature'].values if not dfmin.empty else np.array([]),
            confirmax, confirmin)

def _fill_generic_weather_var(data_tuple, col_name):
    days_raw, values_raw, _, _ = data_tuple
    df = _create_complete_df(days_raw, values_raw, col_name)
    if df.empty: return np.array([]), np.array([]), "NC"
    df[col_name] = df[col_name].fillna(0) # Simple fill with 0 for PRCP/SNOW
    confirm = "C" if not df.empty and df.index[-1].year >= 2020 else "NC"
    return df.index.to_numpy(), df[col_name].values, confirm

def fillSN(data_tuple): return _fill_generic_weather_var(data_tuple, col_name='SN_val')
def fillPRCP(data_tuple): return _fill_generic_weather_var(data_tuple, col_name='PRCP_val')

def avg(data_array): # Unchanged
    if data_array is None or len(data_array) == 0: return np.array([])
    num_periods = len(data_array) // 30
    periods_avg = [np.mean(data_array[i*30:(i+1)*30]) for i in range(num_periods)]
    remainder = len(data_array) % 30
    avg_array_out = np.array(periods_avg)
    if remainder != 0 and len(data_array[-remainder:]) > 0 : # Ensure remainder slice is not empty
        remainder_avg = np.mean(data_array[-remainder:])
        avg_array_out = np.concatenate((avg_array_out, [remainder_avg])) # Ensure avg_array_out is array before concat
    return avg_array_out