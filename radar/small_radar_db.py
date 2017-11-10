# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:13:59 2015

@author: wolfensb
"""

__author__ = 'wolfensb'

import os, numpy as np
import fnmatch
import uuid
from datetime import datetime
from datetime import timedelta
import pyproj as pp
import re
import h5py
import re
import psycopg2 as pg


import warnings
warnings.filterwarnings("ignore")

# USER INPUT
tableName_MXPOL='ltedb_schema.mxpol'
tableName_CH='ltedb_schema.ch_radar'
dbName='ltedb'

FOLDERS_MXPOL=['/ltedata/HYMEX/SOP_2012/Radar/Proc_data/2012/','/ltedata/HYMEX/SOP_2013/Radar/Proc_data/2013/',\
'/ltedata/Davos_2014/Radar/Proc_data/','/ltedata/Davos_2009-2010/MXPOL/Proc_data/','/ltedata/Payerne_2014/Radar/Proc_data/2014/',\
'/ltedata/CLACE2014/Radar/Proc_data/2014/']

FOLDERS_MXPOL_VERT=['/ltedata/HYMEX/SOP_2012/Radar/Proc_data_level2/2012/','/ltedata/HYMEX/SOP_2013/Radar/Proc_data_level2/2013/',\
'/ltedata/Davos_2014/Radar/Proc_data_level2/','/ltedata/Davos_2009-2010/MXPOL/Proc_data_level2/','/ltedata/Payerne_2014/Radar/Proc_data_level2/2014/',\
'/ltedata/CLACE2014/Radar/Proc_data_level2/2014/']

CAMPAIGNS=['Hymex_2012','Hymex_2013','Davos_2014','Davos_2009-2010','Payerne_2014','Clace_2014']


FOLDERS_CH=['/ltedata/MeteoSwiss_Full_Radar_Data_hail/','/ltedata/MeteoSwiss_Full_Radar_Data/',
            '/ltedata/MeteoSwiss_Full_Radar_Data_LowRes/']
ELEVATION_ANGLES_CH=[-0.2,0.4,1,1.6,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,11,13,16,20,25,30,35,40]



class CH_RADAR_db():
    def __init__(self):
        self.db = pg.connect("dbname="+dbName+" host=localhost user=postgres password='Ikot04081415'")
    
        self.cur = self.db.cursor()
        self.cur.execute("SELECT VERSION()")
    def query(self, date=[],radar='',res='H',angle=[], tol_ang=1, tol_time_min=5):
        # Inputs:
            # date = list of 1 or 2 dates in YYYY-mm-dd HH:MM:SS format if len(date) == 1
            # the closest match will be found, if len(date) == 2, all scans with timestamp
            # between date[0] and date[1] will be kept
            # radar = char specifying the radar ("D" = La Dole, "A" = Albis, "L" = Monte Lema, "P" = Plaine Morte)
            # angle = list of 1 or 2 integers, specifying the angle of the scan 
            # (azimuth if RHI, elevation if PPI), if len(angle) == 1, the closest match will be found
            # if len(angle) == 2, all scans with angle inbetween angle[0] and angle[1] will be kept
            # tol_ang == angular tolerance for the closest match (in degrees)
            # tol_time_min == tolerance in time for the closest match (in minutes)
        if np.isscalar(angle):
            angle=[angle]
        sql_query='SELECT * from '+tableName_CH
        if radar!='':
            sql_query+=" WHERE radar='"+radar+"' "
        if len(angle) == 2:
            sql_query+=" AND angle>'" + angle[0] +"' AND date< '" + angle[1]+"' "
        if len(date) == 2:
            sql_query+=" AND date>'" + date[0] + "' AND date<'" + date[1]+"' "
            
        sql_query+=" AND res = '"+res+"' "
        
        # Select by date

        if 'WHERE' not in sql_query:
            sql_query=sql_query.replace('AND','WHERE',1)
        self.cur.execute("SELECT VERSION()")
        self.cur.execute(sql_query)
        x=self.cur.fetchall()

        try:
            if len(angle)==1:
                delta=np.array([abs(q[4]-angle[0]) for q in x])
                idx=np.arange(0,len(delta))
                idx=idx[delta<tol_ang]
                delta=delta[delta<tol_ang]
            
                if len(delta)>0:
                    idx_best=idx[np.where(delta==np.min(delta))]
                    x=[x[i] for i in idx_best]
                else:
                    x=[]
            
            if len(date)==1:

                delta=[abs(q[1]-datetime.strptime(date[0],'%Y-%m-%d %H:%M:%S')) for q in x]
                delta=np.array([d.total_seconds()/60. for d in delta])
                idx=np.arange(0,len(delta))
                
                idx=idx[delta<tol_time_min]

                delta=delta[delta<tol_time_min]

                if len(delta)>0:
                    
                    indexes_best = idx[np.where(delta==np.min(delta))[0]]
                    
                    x = [x[i]  for i in indexes_best]

                else:
                    x=[]
        except:
            x=[]
            
#        x = [xi.rstrip() for xi in x] # Remove white spaces at the end            
        return x

    def populate(self):
          # Create and open the table
        #try:
    
        sql = """CREATE TABLE IF NOT EXISTS %s (
             FILEPATH  CHAR(200) NOT NULL PRIMARY KEY,
             DATE timestamp,
             RADAR CHAR(10),
             RES CHAR(1),
             ANGLE FLOAT)"""%(tableName_CH)
        self.cur.execute(sql)
        # Vertical profiles
        for i,f in enumerate(FOLDERS_CH):
            for root, subFolders, files in os.walk(f):
                for filename in fnmatch.filter(files, 'P*.h5'):
                    print(filename)
                    res = filename[1]
                    
                    if 'A' in filename[2:]:
                        radar='A'
                    elif 'L' in filename[2:]:
                        radar='L'
                    elif 'D' in filename[2:]:
                        radar='D'
                    elif 'P' in filename[2:]:
                        radar='P'
                
                    index_angle=int(re.findall(r"\.([0-9]{3})\.",filename)[0])
                    angle=ELEVATION_ANGLES_CH[index_angle-1] 
                    
                    strdate=re.findall(r"([0-9]{9})",filename)
                    currentTimeString=str(datetime.strptime(strdate[0],'%y%j%H%M'))

                    pathFile=root+'/'+filename  
                        
                    sql = "INSERT INTO %s(FILEPATH,DATE,RADAR,RES,ANGLE) \
                       SELECT '%s', '%s', '%s', '%s','%f' WHERE NOT EXISTS (SELECT FILEPATH FROM %s WHERE FILEPATH = '%s');" % \
                    (tableName_CH, pathFile, currentTimeString, radar,res, angle,
                     tableName_CH, pathFile)
    
                    self.cur.execute(sql)
                    self.db.commit()
                    print currentTimeString
    
        self.db.close()
        

class MXPOL_db():
    def __init__(self):
        self.db = pg.connect("dbname="+dbName+" host=localhost user=postgres password='Ikot04081415'")
    
        self.cur = self.db.cursor()
        self.cur.execute("SELECT VERSION()")
    def query(self, date=[],scan_type='',campaign='', res='L',angle=[], tol_ang=1, tol_time_min=5):
        # Inputs:
            # date = list of 1 or 2 dates in YYYY-mm-dd HH:MM:SS format if len(date) == 1
            # the closest match will be found, if len(date) == 2, all scans with timestamp
            # between date[0] and date[1] will be kept
            # scan_type = string specifying the type of scan ("PPI","RHI", "PPI_SPEC" or "RHI_SPEC"))
            # campaign = one of the MXPOL radar CAMPAIGNS (see top of file)
            # angle = list of 1 or 2 integers, specifying the angle of the scan 
            # (azimuth if RHI, elevation if PPI), if len(angle) == 1, the closest match will be found
            # if len(angle) == 2, all scans with angle inbetween angle[0] and angle[1] will be kept
            # tol_ang == angular tolerance for the closest match (in degrees)
            # tol_time_min == tolerance in time for the closest match (in minutes)
    
        # Create sql
        if np.isscalar(angle):
            angle=[angle]
        sql_query='SELECT * from '+tableName_MXPOL
        if scan_type!='':
            sql_query+=" WHERE scan_type='"+scan_type+"' "
        if campaign!='':
            sql_query+=" AND campaign='"+campaign+"' "
        if len(angle)==2:
            sql_query+=" AND angle>'"+angle[0]+"' AND date< '"+angle[1]+"' "
        if len(date)==2:
            sql_query+=" AND date>'"+date[0]+"' AND date<'"+date[1]+"' "
            
        if 'WHERE' not in sql_query:
            sql_query=sql_query.replace('AND','WHERE',1)
        self.cur.execute("SELECT VERSION()")
        self.cur.execute(sql_query)
        x=self.cur.fetchall()

        try:
            if len(angle)==1:
                delta=np.array([abs(q[4]-angle[0]) for q in x])
                idx=np.arange(0,len(delta))
                idx=idx[delta<tol_ang]
                delta=delta[delta<tol_ang]
                
                if len(delta)>0:
                    idx_best=idx[np.where(delta==np.min(delta))[0]]
                    x=[x[i] for i in idx_best]
                else:
                    x=[]
            
            if len(date)==1:
                delta=[abs(q[2]-datetime.strptime(date[0],'%Y-%m-%d %H:%M:%S')) for q in x]
                delta=np.array([d.total_seconds()/60. for d in delta])
                idx=np.arange(0,len(delta))

                idx=idx[delta<tol_time_min]
                delta=delta[delta<tol_time_min]


                if len(delta)>0:
                    indexes_best = idx[np.where(delta==np.min(delta))[0]]
                    x = [x[i]  for i in indexes_best]

                else:
                    x=[]
        except:
            x=[]
        
        return x


    def populate(self):
          # Create and open the table
        #try:
    
        sql = """CREATE TABLE IF NOT EXISTS %s (
             FILEPATH  CHAR(200) NOT NULL PRIMARY KEY,
             CAMPAIGN CHAR(30),
             DATE timestamp,
             SCAN_TYPE CHAR(10),
             ANGLE FLOAT)"""%(tableName_MXPOL)
        self.cur.execute(sql)
       
        for i,f in enumerate(FOLDERS_MXPOL):
            for root, subFolders, files in os.walk(f):
                print(root)
                for filename in fnmatch.filter(files, 'MXPol-polar-*.nc'):
                    print(filename)
                    if 'RHI' in filename:
                        scan_type='RHI'
                    elif 'PPI' in filename:
                        scan_type='PPI'
                    if 'SPEC' in filename:
                        scan_type+='_SPEC'
                    strdate=re.findall(r"([0-9]{8}-[0-9]{6})",filename)
                    currentTimeString=str(datetime.strptime(strdate[0],'%Y%m%d-%H%M%S'))
                    angle=re.findall(r"-([0-9]{3})_",filename)[0]
                    angle=int(angle)
                    pathFile=root+'/'+filename  

                    sql = "INSERT INTO %s(FILEPATH,CAMPAIGN,DATE,SCAN_TYPE,ANGLE) \
                       SELECT '%s', '%s', '%s', '%s','%f' WHERE NOT EXISTS (SELECT FILEPATH FROM %s WHERE FILEPATH = '%s')" % \
                    (tableName_MXPOL, pathFile, CAMPAIGNS[i], currentTimeString, scan_type, angle,
                     tableName_MXPOL,pathFile)
                    
                    self.cur.execute(sql)
                    self.db.commit()
        # Vertical profiles            
        for i,f in enumerate(FOLDERS_MXPOL_VERT):
            for root, subFolders, files in os.walk(f):
                print(root)
                for filename in fnmatch.filter(files, 'MXPol-profile-*.nc'):
                    strdate=re.findall(r"([0-9]{8}-[0-9]{6})",filename)
                    currentTimeString=str(datetime.strptime(strdate[0],'%Y%m%d-%H%M%S'))

                    scan_type='V_DOPPLER'
                    angle = int(90)
                    pathFile=root+'/'+filename  
                        
                    sql = "INSERT INTO %s(FILEPATH,CAMPAIGN,DATE,SCAN_TYPE,ANGLE) \
                       SELECT '%s', '%s', '%s', '%s','%f' WHERE NOT EXISTS (SELECT FILEPATH FROM %s WHERE FILEPATH = '%s')" % \
                    (tableName_MXPOL, pathFile, CAMPAIGNS[i], currentTimeString, scan_type, angle,
                     tableName_MXPOL,pathFile)
                    self.cur.execute(sql)
                    self.db.commit()
                    
        self.db.close()
        

# date=['2014-05-13 00:00:00']
if __name__=='__main__':
    d=CH_RADAR_db()
#    p = d.query(date=['2015-08-13 12:00:00'],radar='L', angle=1)
#    print(p)
    
    d.populate()
#    q=d.query(date=['2014-04-08 02:00:00'],campaign='Payerne_2014',tol_time_min=5,scan_type='PPI',angle=[5],tol_ang=5)
#    print q
#    print len(q)


