#!/usr/bin/env python
# coding: utf-8

"""
Module SJM Anonymization Script
"""

__author__ = "Ishaan, Felix"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse

import pandas as pd
import time
import datetime
import random
from Cryptodome.Hash import MD5, HMAC

def main(args):
    """ Main entry point of the app """
    print("Anonymizing ...")
    print(args)

    file_loc = args.file
    origdata = pd.read_csv(file_loc, sep=',', low_memory=False)

    print(origdata)

    print(args.col_customer)

    ID_field = args.col_customer    #TBU CustomerID field name
    TS_field = args.col_time        #TBU TS field name 

    sub = origdata.loc[:,:] #Uncomment for applying on complete dataset

    def anonymizeID(data,ID_field):
        df=pd.DataFrame(data.loc[:,ID_field])
        df2=df.applymap(lambda x: ((HMAC.new(b"key", bytes(x), MD5)).hexdigest()))
        return df2.loc[:,ID_field]

    def shiftTimeForTrace(data,TS_field):
    #     print(df.head())
        df=pd.DataFrame(data.loc[:,TS_field])
        df2 = df.loc[:,TS_field].apply(lambda x: pd.to_datetime(x)) 
    #     print(df2.head())
        rand_days = random.randint(-5,5) #range can be updated
        df2 = df2 + pd.DateOffset(days=rand_days)
    #     print(df2.head())
        return df2

    #OG subset for reference
    print(sub.head())

    # sub1=sub.sort_values(by=ID_field)
    sub1 = sub.loc[:,:]
    uniqueIDs = list(sub[ID_field].unique())
    # sub2 = sub.loc[:,TS_field].apply(lambda x: pd.to_datetime(x))
    for ID in uniqueIDs:
        sub3=sub1.loc[sub1[ID_field] == ID]
        sub3.loc[:,TS_field]=shiftTimeForTrace(sub3,TS_field)
    #     print(sub1.loc[sub1[ID_field] == ID][TS_field])
    #     print(pd.DataFrame(sub3[TS_field]))
        sub1.loc[sub1[ID_field] == ID,TS_field] = pd.DataFrame(sub3[TS_field])

    # Results post TS shift
    print(sub1.head())

    sub4 = sub1.loc[:,:]
    sub4.loc[:,ID_field] = anonymizeID(sub4,ID_field)

    #Results post ID anonymization
    print(sub4.head())
    sub4.to_csv('out.csv',index=False)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Source file (CSV)")
    
    parser.add_argument("--col_customer", default = "CustomerID")
    parser.add_argument("--col_time", default = "TIMESTAMP")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)



