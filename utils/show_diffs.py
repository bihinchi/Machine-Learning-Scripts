import pandas as pd
import os
import shutil



def get_differences_dataframe(data, data_old):
    dl = pd.concat([data, data_old]).drop_duplicates(keep=False)
    dr = pd.concat([data_old, data]).drop_duplicates(keep=False)
    diffs = pd.concat([dl, dr], axis=1, ignore_index=True)
    diffs = diffs.reset_index().drop_duplicates(subset="index")
    diffs = diffs.drop(2, 1).drop(0,1)
    diffs = diffs.rename(columns={1 : "New", 3 : "Old"})
    return diffs



def get_file_name(id):
    nums = len(str(id))
    zeros = "0" * (6 - nums)
    return zeros + str(id) + ".jpg"





def process(new, old, source, destination):
    diffs = get_differences_dataframe(new, old)
    print("Found {} differences".format(diffs.size))
    diffs.to_csv("diffs.csv", index=False)
    print("Saved to diffs.csv")

    if destination == None:
        destination = "image_diffs"
        try:
            os.mkdir(destination)
        except Exception:
            pass
        

    ids = diffs.index.to_frame()[0]
    files = ids.apply(get_file_name)
    
    print("Copying files")
    for file in files:
        name = os.path.join( source, file )
        if os.path.isfile( name ) :
            try:
                shutil.copy( name, destination)
            except Exception as e:
                print(e)
        else :
            print('file does not exist', name)
    

    print("Done!")



'''

'''


if __name__ == '__main__':
    import argparse, codecs
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--tpath', help='path to the directory with test picture')
    arg_parser.add_argument('--dpath', help='path to store pictures that were predicted differently (default: save in folder locally)')
    arg_parser.add_argument('--new_result', help='csv file with new results')
    arg_parser.add_argument('--old_result', help='csv file with old results')

    ns = arg_parser.parse_args()

    if ns.tpath is None or ns.new_result is None or ns.old_result is None:
        arg_parser.print_help()
    else:
        data = pd.read_csv(ns.new_result)
        data_old = pd.read_csv(ns.old_result)

        process(data, data_old, ns.tpath, ns.dpath)
