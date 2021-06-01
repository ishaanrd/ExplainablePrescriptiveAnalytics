#!/usr/bin/env python
# coding: utf-8

"""
Module SJM CSV2EventLog Parser Script
"""

__author__ = "Ishaan, Felix"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse

import pandas as pd
import re


def main(args):
    """ Main entry point of the app """
    print("Parsing CSV ...")
    print(args)

    file_loc = args.file
    mode = args.mode
    origdata = pd.read_csv(file_loc, sep=';', index_col=False)

    #    print(origdata)

    print(args.col_customer)

    ID_field = args.col_customer  # TBU CustomerID field name
    uniqueIDs = list(origdata[ID_field].unique())

    column_names = ["timestamp", "case_id", "event", "event_instance", "status", "resource"]
    event_log = pd.DataFrame(columns=column_names)
    act_inst = 1
    f = 0

    # Splitting Message field into multiple fields
    origdata['Event0'] = origdata['Message'].apply(str).str.partition()[0]
    origdata['Event1'] = origdata['Message'].apply(str).str.partition()[2].str.partition()[0]
    origdata['Event2'] = origdata['Message'].apply(str).str.partition()[2].str.partition()[2]
    origdata['Event3'] = origdata['Event2'].apply(str).str.partition()[0].str.partition()[0]

    cols = ['Event0', 'Event1']
    origdata['Event'] = origdata[cols].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
    # Special handling for TaskEvent in Message field
    origdata.loc[origdata['Event'] == 'Taskevent:', 'Event'] = origdata['Event3']

    def ro_generic(df, eve_inst):  # with task enumeration
        cnt_TE = 0
        #     eve_inst = 1
        #     last_event
        for index, row in df.iterrows():
            curr_event = df.loc[index, 'Event']
            if cnt_TE == 0:
                item_cnt = 0
                inc_inst = 999
                task_num = 1
            if curr_event == 'taskDownloaded,':
                cnt_TE += 1
                msg = df.loc[index, 'Message']
                item = re.search('item (.+?), timeUsed', msg).group(1)
                #             print(msg[-4:])
                if item == "0":
                    if cnt_TE == 1:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'WelcomeMsg'
                        event_log.loc[index, 'resource'] = 'System'
                        event_log.loc[index, 'status'] = 'start'
                        event_log.loc[index, 'event_instance'] = eve_inst
                        inc_inst = eve_inst
                        eve_inst += 1
                    elif cnt_TE == 2:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'WelcomeMsg'
                        event_log.loc[index, 'resource'] = 'System'
                        event_log.loc[index, 'status'] = 'complete'
                        event_log.loc[index, 'event_instance'] = inc_inst
                        inc_inst = 999
                    #                     eve_inst +=1
                    prev_item = "0"
                else:
                    if item != prev_item: item_cnt = 0
                    item_cnt += 1
                    prev_item = item
                    if item_cnt == 3:
                        #                     print("Im here",item_cnt, index)
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'CodingTask_' + str(task_num)
                        event_log.loc[index, 'resource'] = 'User'
                        event_log.loc[index, 'status'] = 'start'
                        event_log.loc[index, 'event_instance'] = eve_inst
                        inc_inst = eve_inst
                        eve_inst += 1
                    elif item_cnt == 4:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'CodingTask_' + str(task_num)
                        event_log.loc[index, 'resource'] = 'User'
                        event_log.loc[index, 'status'] = 'complete'
                        event_log.loc[index, 'event_instance'] = inc_inst
                        task_num += 1
                        inc_inst = 999
            #                     eve_inst +=1
            else:
                event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                event_log.loc[index, 'event'] = df.loc[index, 'Event']
                event_log.loc[index, 'status'] = 'complete'
                event_log.loc[index, 'event_instance'] = eve_inst
                eve_inst += 1
                if df.loc[index, 'Event'] == "Manualscores":
                    event_log.loc[index, 'resource'] = 'Admin'
                else:
                    event_log.loc[index, 'resource'] = 'System'
        return eve_inst + 1

    def ro_specific(df, eve_inst):  # RowiseOperation, task name with task number
        cnt_TE = 0
        #     eve_inst = 1
        #     last_event
        for index, row in df.iterrows():
            curr_event = df.loc[index, 'Event']
            # print(curr_event)
            if cnt_TE == 0:
                item_cnt = 0
                inc_inst = 999
            if curr_event == 'taskDownloaded,':
                cnt_TE += 1
                msg = df.loc[index, 'Message']
                item = re.search('item (.+?), timeUsed', msg).group(1)
                #             print(msg[-4:])
                if item == "0":
                    if cnt_TE == 1:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'WelcomeMsg'
                        event_log.loc[index, 'resource'] = 'System'
                        event_log.loc[index, 'status'] = 'start'
                        event_log.loc[index, 'event_instance'] = eve_inst
                        inc_inst = eve_inst
                        eve_inst += 1
                    elif cnt_TE == 2:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'WelcomeMsg'
                        event_log.loc[index, 'resource'] = 'System'
                        event_log.loc[index, 'status'] = 'complete'
                        event_log.loc[index, 'event_instance'] = inc_inst
                        inc_inst = 999
                    #                     eve_inst +=1
                    prev_item = "0"
                else:
                    if item != prev_item: item_cnt = 0
                    item_cnt += 1
                    prev_item = item
                    if item_cnt == 3:
                        #                     print("Im here",item_cnt, index)
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'CodingTask_' + item
                        event_log.loc[index, 'resource'] = 'User'
                        event_log.loc[index, 'status'] = 'start'
                        event_log.loc[index, 'event_instance'] = eve_inst
                        inc_inst = eve_inst
                        eve_inst += 1
                    elif item_cnt == 4:
                        event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                        event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                        event_log.loc[index, 'event'] = 'CodingTask_' + item
                        event_log.loc[index, 'resource'] = 'User'
                        event_log.loc[index, 'status'] = 'complete'
                        event_log.loc[index, 'event_instance'] = inc_inst
                        inc_inst = 999
            #                     eve_inst +=1
            else:
                event_log.loc[index, 'timestamp'] = df.loc[index, 'Timestamp']
                event_log.loc[index, 'case_id'] = df.loc[index, 'Developer ID']
                event_log.loc[index, 'event'] = df.loc[index, 'Event']
                event_log.loc[index, 'status'] = 'complete'
                event_log.loc[index, 'event_instance'] = eve_inst
                eve_inst += 1
                if df.loc[index, 'Event'] == "Manualscores":
                    event_log.loc[index, 'resource'] = 'Admin'
                else:
                    event_log.loc[index, 'resource'] = 'System'
        return eve_inst + 1
        #     return event_log

    for ID in uniqueIDs:
        user_el = origdata.loc[origdata['Developer ID'] == ID]
        #     event_log = event_log.append(ro(user_el))
        if mode == "specific":
            if f == 0:
                act_inst = ro_specific(user_el, 1)
                f = 1
            else:
                act_inst = ro_specific(user_el, act_inst)
        elif mode == "generic":
            if f == 0:
                act_inst = ro_generic(user_el, 1)
                f = 1
            else:
                act_inst = ro_generic(user_el, act_inst)

    print("Exporting reformated csv for R", event_log['event'])
    event_log.to_csv('log4bupar.csv', index=False)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Source file (CSV)")

    parser.add_argument("mode", help="generic/specific")

    parser.add_argument("--col_customer", default="Developer ID")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)



