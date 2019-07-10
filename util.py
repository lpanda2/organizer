"""
General I/O or useful Utility Functions
"""
import json
import time
import datetime
import hashlib
import subprocess
import logging
import logging.config
import os
from functools import wraps, partial
import re
import configparser
import pandas as pd
import numpy as np
import boto3
import yaml
import shutil
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
from sqlalchemy import create_engine

from botocore.client import Config
from s3fs import S3FileSystem
from s3fs.core import split_path
from botocore.exceptions import ClientError

from pkg_resources import resource_filename

## -------------------------------------------------------------------------- ##
## Case conversion and renames
## -------------------------------------------------------------------------- ##

CAMEL = lambda x, y: x + y[0].title() + y[1:]

def camel_converter(data):
    """
    Convert the columns in a dataframe to camelCase.
    @data: dataframe
    returns: df with revised column names
    """
    match = r'(-[a-z])|([A-Z])'
    camel = lambda x: x.group().replace("-", "").title()
    cm_converter = np.vectorize(lambda s: re.sub(match, camel, s))
    data.columns = cm_converter(data.columns)
    return data


def replace_case(to_case, val):
    """
    Using a match on a snake_case, camelCase or kebab-case column,
    converts any of the above into the desired output case.
    Returns the input if the input case is the same as the output case.
    """
    mc = r'[A-Z]|-([a-z])| |_([a-z])'
    sk = lambda x: '_' + x.group(0).strip(' ').strip('.').strip('-').lower()
    cm = lambda x: x.group(0).upper().strip('_').strip('.').strip('-')
    kb = lambda x: '-' + x.group(0).lower().strip(' ').strip('.').strip('_')
    cases = {'snake': lambda s: re.sub(mc, sk, s).replace('__', '_'),
             'camel': lambda s: re.sub(mc, cm, s),
             'kebab': lambda s: re.sub(mc, kb, s).replace('--', '-')}
    return cases[to_case](val)


def convert_case(data, case=''):
    """
    Converts all column names in a dataframe to the case specified.
    @data: pd.DataFrame or list of pd.DataFrames
    @case: str Available options are 'snake', 'camel', and 'kebab'.
    """
    data.columns = map(partial(replace_case,case), data.columns)
    return data


## -------------------------------------------------------------------------- ##
## I/O Functions
## -------------------------------------------------------------------------- ##

def read_yaml(path):
    """
    Read a path and if from s3, load the data.
    @path: str
    returns: dict, parsed yaml file of data
    """
    with open(path, "rb") as yfile:
        return yaml.load(yfile)


def read_json(path, s3_bucket=None):
    """
    Read a path and if from s3, load the data.
    @path: str
    @s3_bucket: str, bucket_name if there
    returns: dict, parsed json file of data
    """
    if s3_bucket:
        client_config_read = Config(connect_timeout=70,
                                    read_timeout=120)
        client = boto3.client("s3", config=client_config_read)
        path = '/'.join(path.split('/')[1:]) if s3_bucket in path else path
        obj = client.get_object(Bucket=s3_bucket, Key=path)["Body"].read().decode()
        return json.loads(obj)
    else:
        with open(path, "rb") as data_file:
            return json.load(data_file)

def isoconverter(o):
    """Converts dates to string dates"""
    return o.strftime("%Y-%m-%d") if isinstance(o, datetime.datetime) else o


def dict_dump(dict_):
    """Converts a dictionary into a string for JSON file dumping"""
    return json.dumps(dict_, indent=4, default=isoconverter).encode()


def write_json(path, data, s3_bucket=None, kwargs={}):
    """
    Take in a path, s3 or disk, and dump the JSON data into that path.
    @path: str, key for the s3 bucket (should not have bucket name in it)
    @data: dict, to be dumped into json
    @s3_bucket: str, bucket name
    @kwargs: dict, serverside encryption if needed
    """
    if s3_bucket:
        s3b = boto3.resource('s3').Bucket(s3_bucket)
        data = dict_dump(data) if (isinstance(data, dict) or
                                   isinstance(data, list)) else data
        path = '/'.join(path.split('/')[1:]) if s3_bucket in path else path
        s3b.Object(key=path).put(Body=data, **kwargs)
    else:
        with open(path, 'wb') as data_file:
            data_file.write(dict_dump(data))


def write_parquet(path, data, s3_bucket=None, kwargs={}):
    """
    Take in a path, s3 or disk, and dump the JSON data into that path.
    @path: str, key for the s3 bucket (should not have bucket name in it)
    @data: dict, to be dumped into json
    @s3_bucket: str, bucket name
    @kwargs: dict, serverside encryption if needed
    """
    if s3_bucket:
        s3b = boto3.resource('s3').Bucket(s3_bucket)
        path = '/'.join(path.split('/')[1:]) if s3_bucket in path else path
        s3b.Object(key=path).put(Body=data, **kwargs)
    else:
        data.to_parquet(path)


def decode_df(data):
    """
    Fix an issue when reading in a dataframe in Python 3
    and all the
    unicode strings get converted to a byte string.
    @data: df
    returns: df, dataframe where byte converted to str
    """
    decode = lambda s: s.decode() if isinstance(s, bytes) else str(s)
    decode_column = lambda arr: [decode(string) for string in arr]
    decode_these = [x for x in data.columns if data[x].dtype == object]
    for col in decode_these:
        data[col] = decode_column(data[col].values)
    rename = {old: decode(old) for old in data}
    data = data.rename(columns=rename)
    return data


def read_parquet(path):
    """
    String Enforced Parquet Reader -- Wrapper for PD function.
    @path: str, link to path
    """
    data = decode_df(pd.read_parquet(path))
    data = data.fillna(np.nan).replace({'None': np.nan})
    return data


def write_excel(data, writer, sheet):
    """Write data to an """
    data.to_excel(writer, index=False, sheet_name=sheet)
    worksheet = writer.sheets[sheet]
    worksheet.set_zoom(85)


def append_to_csv(data, path, fname):
    """
    Appends data to a CSV file if that file exists in a given
    directory.
    @table: pd.DataFrame, some tabular data
    @path: str, directory path
    @fname: str, filename with extension in path directory.
    """
    if fname in os.listdir(path):
        data.to_csv(path + fname, mode='a', header=False, index=False)
    else:
        data.to_csv(path + fname, index=False)


def read_some_bins(bins):
    """ Read a subset of bin paths.
    @bins: list, list of string paths to bins"""
    return pd.concat([read_parquet(b) for b in bins], ignore_index=True)


def read_file(path):
    """ Reads csvs, parquets, h5s, and excel files."""
    if '.csv' in path:
        try:
            data = pd.read_csv(path, dtype=str, nrows=1000)
        except:
            try:
                data = pd.read_csv(path, dtype=str, encoding='latin1', nrows=1000)
            except:
                try:
                    data = pd.read_csv(path, dtype=str, sep='\t')
                except:
                    raise Exception(path, '-----data import failed------')
    elif '.parquet' in path:
        data = read_parquet(path)
    elif '.h5' in path:
        data = pd.read_hdf(path)
    elif '.xlsx' in path:
        data = pd.read_excel(path, dtype=str, skiprows=2, encoding='latin1')
    return data

## -------------------------------------------------------------------------- ##
## Configuration Creation
## -------------------------------------------------------------------------- ##


def get_default_configs():
    """ Get default configurations for os. """
    default_filename = ('posix_configs.ini' if os.name == 'posix'
                        else 'win_configs.ini')
    default_filepath = resource_filename(__name__, default_filename)
    return default_filepath


def get_user_configs():
    """ Get user specific configurations for os. """
    home = os.path.expanduser("~")
    return [home + "/.remedy.ini"]


def as_dict(config):
    """
    Converts a ConfigParser object into a dictionary.
    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    the_dict = {}
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict

def load_configs():
    """Load configuration file and user specific settings"""
    code_config = read_yaml('additional_config.yaml')
    parser = configparser.ConfigParser()
    config_file_paths = get_user_configs()
    default_filename = get_default_configs()
    parser.read(default_filename)
    res = parser.read(config_file_paths)
    parser = as_dict(parser)
    if res:
        print ("Additional config files loaded: {}".format(res))
    code_config.update(parser)
    return code_config

def flatten_nest(d):
    """
    Flattens dictionary being used as PAYER_CONFIG variable and saves
    to output location with timestamp
    """

    # Lists of keys and values to be zipeed in flat dict before returning
    keys_ = []
    values_ = []

    # Iterate over items and do subtour of dict values
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_nest(v)
        else:
            print(str(type(v)))
            if 'function' in str(type(k)):
                k_new = str(k.__name__)
                keys_.append(k_new)
            else:
                keys_.append(k)
            if 'function' in str(type(v)):
                v_new = str(v.__name__)
                values_.append(v_new)
            elif 'list' in str(type(v)):
                if any('function' in str(type(L)) for L in v):
                    v_new = ", ".join(str(l.__name__) for l in v)
                    values_.append(v_new)
                else:
                    v_new = ", ".join(l for l in v)
                    values_.append(v_new)
            else:
                values_.append(v)
    assert len(keys_) == len(values_)
    return dict(zip(keys_, values_))

def log_configs(payer, state, payer_config, module_config):
    """
    Log / save payer-specific configurations as well as module configurations
        1) payer_configuration.py
        2) configuration.yaml
    """

    # Save location
    log_config_location = module_config.get('drives').get('m-drive') + \
                          "/run_config_logs/"
    if not os.path.exists(log_config_location):
        os.makedirs(log_config_location)

    # ---- ---- ----
    # Module wide config
    write_json(
        path=log_config_location + "{}_{}_{}_CONFIG.csv".format(
            payer, state,
            datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        ),
        data=module_config
    )

    # Payer wide config
    path=log_config_location + "{}_{}_{}_PAYER_CONFIG.json".format(
        payer, state,
        datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    )
    payer_config_ = flatten_nest(payer_config.get(payer).get(state))
    with open(path, "w") as fp:
        json.dump(payer_config_, fp)

    # ---- ---- ----
    # Data Dictionary
    shutil.copy2(
        CONFIG.get('drives').get('m-drive') +
        "/data_dicts/{}_{}_data_dictionary.csv".format(payer, state),
        os.path.join(
            log_config_location,
            "{}_{}_{}_data_dictionary.csv".format(
                payer, state,
                datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            )
        )
    )

CONFIG = load_configs()


## -------------------------------------------------------------------------- ##
## SQL Engine Creation
## -------------------------------------------------------------------------- ##

def remedy_sql_connection():
    """Create a sql engine connection to Remedy's
    SQL databases"""
    conn = CONFIG.get('sql-engine-prod')
    conn = conn['server'].format(u=conn['user'],
                                 p=conn['password'])
    sqle = create_engine(conn)
    return sqle

def epdev_sql_connection():
    """ Create a sql engine connection to the post gres
    epdev schema."""
    conn = CONFIG.get('sql-engine-epdev')
    conn = conn['server'].format(u=conn['user'], p=conn['password'])
    sqle = create_engine(conn)
    return sqle

def mercator_sql_connection():
    """Create a sql engine connection to mercator."""
    #conn = CONFIG.get('sql-engine-mercator')
    #conn = conn['server'].format(u=conn['user'], p=conn['password'])
    conn = 'postgresql://lpanda@0.0.0.0:5432/mercator' # local
    sqle = create_engine(conn)
    return sqle


## -------------------------------------------------------------------------- ##
## Decorators
## -------------------------------------------------------------------------- ##

def time_this(func):
    """
    Time decorator to time a function
    @func: function
    """
    custom_setup_logging(level=logging.ERROR)
    LOG = logging.getLogger()
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper for function that prints time to stdout"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        LOG.info("{} takes {} seconds.".format(func.__name__, round(end-start, 2)))
        return result
    return wrapper


def pass_thru(column):
    """
    Decorator to skip the function if the column
    it creates is already there.
    @column: str, column name in first arg to skip
    """
    def decorated(func):
        def wrapper(*args, **kwargs):
            if column in args[0]:
                return args[0]
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorated


def avoid_null(column):
    """
    Decorator to avoid pre-processing functions calling ".str" accessor
    of columns that are entirely null (thus not determined to be str on read in)
    @func: function to be decorated
    """
    def decorated(func):
        def chance_the(*args, **kwargs):
            if args[0][column].isnull().sum() == args[0].shape[0]:
                return args[0]
            else:
                return func(*args, **kwargs)
        return chance_the
    return decorated


## -------------------------------------------------------------------------- ##
##  OS and File System
## -------------------------------------------------------------------------- ##

def make_dirs(directory):
    """Makes a directory given a dir path."""
    if not os.path.isdir(directory):
        os.makedirs(directory)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])\
                     .strip().decode()

def branch():
    return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])\
                     .strip().decode()


def get_last_time(save_dir):
    """ Gets last run time in log file."""
    if 'log.txt' in os.listdir(save_dir):
        with open(save_dir + 'log.txt', 'r') as f:
            data = f.readlines()
        data = [x[:16] for x in data if 'reason' in x]
        return data[-1]
    else:
        return now()


def archive_old_files(save_dir, timed=False):
    """Archives all old files within a directory if Log is not in the filename."""
    files = [x for x in os.listdir(save_dir) if ('log' not in x) and ('archived' not in x)]
    old_paths = [save_dir + x for x in files]
    full_path = (save_dir + 'archived/{}/'.format(get_last_time(save_dir))
                 if timed else save_dir + 'archived/')
    make_dirs(full_path)
    new_files = [full_path + x for x in files]
    for i, x in enumerate(old_paths):
        os.rename(x, new_files[i])



## -------------------------------------------------------------------------- ##
## Logging information
## -------------------------------------------------------------------------- ##

def now(fmt='%Y-%m-%d-%H-%M'):
    """Returns current datetime in isoformat."""
    return datetime.datetime.now().strftime(fmt)


def now_to_log(dired, msg=None):
    """Write last updated in directory"""
    with open(dired + 'Last Updated Timestamp.txt'.format(now()), 'w') as file_:
        file_.write(now())
        if msg:
            file_.write('\r\n\r\n{}'.format(msg))


def write_status_to_log(dired, msg=None):
    msg = msg if msg else "No Reason Provided"
    with open(dired + 'log.txt', 'a') as log_file:
        update_msg = '\n' + now() + ' \t | reason -- {}\n'.format(msg)
        log_file.write(update_msg)
        log_file.write('Used following commit: \n\t{}\t{}\n'.format(
            branch(), get_git_revision_short_hash()))



## -------------------------------------------------------------------------- ##
## Hashing Functions
## -------------------------------------------------------------------------- ##

get_bin = lambda x: x.split("/")[-1].split(".")[0].split("_")[1]
get_bins = np.vectorize(get_bin)
make_bin = lambda s: hashlib.md5(s).hexdigest()[:3] if s == s else s
make_bins = np.vectorize(make_bin)
hash_mems = np.vectorize(lambda s: hashlib.md5(s).hexdigest() if s == s else s)



## -------------------------------------------------------------------------- ##
##  Statistical Functions
## -------------------------------------------------------------------------- ##


def deciles():
    """Returns a list of [0.1 .. 0.9]"""
    return [round(x*.1, 2) for x in range(0, 11)]

def ventiles():
    """Returns a list of [0.05 - 0.95]"""
    return [round(x*.05, 2) for x in range(0, 21)]

def quintiles():
    """List of [0.2, 0.4, 0.6, 0.8]"""
    return [round(x*0.2,2) for x in range(0,6)]

def quantiles():
    """List of [0.25, 0.5, 0.75]"""
    return [round(x*0.25, 2) for x in range(0,5)]

def default_perc():
    return [0.5]

PERCS = {
    'decile': deciles(),
    'quantile': quantiles(),
    'quintile': quintiles(),
    'ventile': ventiles(),
    'default': default_perc()
}

PERIOD = {
    'triggers': ['triggerFromDate', 'triggerThruDate'],
    'episode': ['lookBackDate', 'lookForwardDate'],
    'default': ['lookBackDate', 'triggerFromDate', 'triggerThruDate', 'lookForwardDate']
}

BUNDLE_FIELDS = {
    'name': 'bundleName',
    'id': 'bundleDefinitionId',
    'cat': 'bundleCategory',
    'scat': 'bundleSubCategory',
    'default': 'bundleDefinitionId'
}


## -------------------------------------------------------------------------- ##
## Tables and Nice Distributions
## -------------------------------------------------------------------------- ##

def get_dtypes_and_nulls(data):
    """Get DataTypes & Nulls For Data"""
    return pd.concat([
        data.dtypes.sort_index().to_frame(),
        data.isnull().sum().to_frame(),
        (data.isnull().sum().to_frame() / data.shape[0])
    ], axis=1, keys=['Dtype', 'Null Obs', 'Percent Null'])


def value_counts(col):
    """ Get Value Counts for a column"""
    return pd.concat([col.value_counts(), col.value_counts(normalize=True)],
                     axis=1, keys=['Count', 'Percent'])

def describe(cols, percentiles=[.25, 0.5, 0.75]):
    """ Describe a series of columns and stack distribs together. """
    return pd.concat([col.describe(percentiles=percentiles) for col in cols], axis=0,
                     keys=[col.name for col in cols])

def crosstabs(col1, col2, normalize='all', margins=False):
    """Generate a crosstab with Percent info & Count info together"""
    return pd.concat([
        pd.crosstab(col1, col2, margins=margins),
        pd.crosstab(col1, col2, normalize=normalize, margins=margins)
    ], axis=1, keys=['Counts', 'Percent of {}'.format(normalize.title())])


def isin(str_, list_):
    """ Get is in with list"""
    return any([x in str_ for x in list_])

def percent_change(data, old, new, create_col=False):
    """Get Percent Change Between Two Columns"""
    if create_col:
        new_name = '{}_{}_per_change'.format(old, new)
        data[new_name] = (data[new] - data[old]) / data[old]
        return data
    else:
        return (data[new] - data[old]) / data[old]


def volume_delta_by_year(data, field='id'):
    """Volume and Cost Change By Bundle-Year"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    data.year = data.year.astype(str)
    table = pd.pivot_table(
        data,
        columns='year',
        index=bundle,
        values=['episodeId'],
        aggfunc=len
    )
    years = data.year.unique()
    years.sort()
    for i in range(len((years))):
        if i+1 < len(years):
            create = ('episodeId', '{}-{} %'.format(years[i], years[i+1]))
            old = ('episodeId', years[i])
            new = ('episodeId', years[i+1])
            table = percent_bet_cols(table, create, old, new)
    table = table.sort_index(axis=1)
    return table


def volume_delta_by_quarter(data, field='id'):
    """Volume and Cost Change By Bundle-Quarter"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    table = pd.pivot_table(
        data,
        columns='quarter',
        index=bundle,
        values=['episodeId'],
        aggfunc=len
    )
    quarters = data.quarter.unique()
    quarters.sort()
    for i in range(len(quarters)):
        if i+1 < len(quarters):
            create = ('episodeId', '{}-{} %'.format(quarters[i], quarters[i+1]))
            old = ('episodeId', quarters[i])
            new = ('episodeId', quarters[i+1])
            table = percent_bet_cols(table, create, old, new)
    table = table.sort_index(axis=1)
    return table


def episode_window(data, percentile='decile', field='default'):
    """Episode Window By Bundle"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    agg_cols = [bundle]
    pcs = PERCS.get(percentile, PERCS.get('default'))
    table = data.groupby(agg_cols)['episodeWindowDays'].describe(percentiles=pcs)
    return table


def trigger_window(data, percentile='decile', field='default'):
    """Trigger Window By Bundle"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    agg_cols = [bundle, 'siteOfCare']
    pcs = PERCS.get(percentile, PERCS.get('default'))
    table = data.groupby(agg_cols)['triggerWindowDays'].describe(percentiles=pcs)
    return table


def episode_cost(data, percentile='decile', cost='episodeAllowedAmount', field='default'):
    """Episode Cost (default Allowed) By Bundle"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    agg_cols = [bundle]
    pcs = PERCS.get(percentile, PERCS.get('default'))
    table = data.groupby(agg_cols)[cost].describe(percentiles=pcs)
    return table


def episode_dates(data, period='triggers', field='default'):
    """Episode Date Min & Maxes"""
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    agg_cols = [bundle]
    per = PERIOD.get(period, PERIOD.get('default'))
    table = data.groupby(agg_cols)[per].describe()
    return table


def episode_ages(data, field='default'):
    bundle = BUNDLE_FIELDS.get(field, BUNDLE_FIELDS.get('default'))
    agg_cols = [bundle]
    table = data.groupby(agg_cols)['memberAge'].describe()
    return table


def compare_dfs(A, B, name_a='', name_b='', log='/tmp/log.txt'):
    """ Compare two dataframes, A & B. """
    if os.path.exists(log):
        os.remove(log)

    with open(log, 'w') as f:

        f.write('Comparing {} and {}'.format(name_a, name_b))
        f.write(now())
        f.write('--------------------------------------')

        # compare columns
        f.write('Columns in {} but not in {}'.format(name_a, name_b))
        f.write(str(set(A.columns).difference(set(B.columns))))
        f.write('\n')

        f.write('Columns in {} but not in {}'.format(name_b, name_a))
        f.write(str(set(B.columns).difference(set(A.columns))))
        f.write('\n')

        f.write('Columns in both')
        in_both = set(A.columns).intersection(set(B.columns))
        f.write('Len: {}'.format(len(in_both)))
        if len(in_both) < 10:
            f.write('\tColumns are {}'.format(in_both))
            f.write('\n')

        # compare data types
        f.write('Dtype compare')
        pd.set_option('display.max_rows', 150)
        dtypes = pd.concat([A.dtypes.sort_index(), B.dtypes.sort_index()],
                           keys=['A', 'B'], axis=1)
        f.write(str(dtypes))
        f.write('\n')
        f.write('\n')

        # select datatypes for columns in both
        floats = dtypes.loc[(dtypes == 'float64').all(axis=1)].index
        ints = dtypes.loc[(dtypes == 'int64').all(axis=1)].index
        objs = dtypes.loc[(dtypes == 'object').all(axis=1)].index

        # compare rows after merging
        merged = pd.merge(A, B, on=['episode_id', 'bpid'], how='outer',
                          validate='1:1', indicator=True, suffixes=[name_a, name_b])
        f.write('------------------------------')
        f.write('Rows')
        f.write('------------------------------')
        f.write('Shapes: {} vs. {}'.format(A.shape, B.shape))
        f.write(str(merged._merge.value_counts(normalize=True)))
        f.write('\n\n')

        # subset to rows in both but spit out others:
        unmerged = merged.loc[merged._merge != 'both']
        C = merged.loc[merged._merge == 'both']

        # compare overall values
        final = {}
        for col in floats + ints:
            final[col] = (1 - np.isclose(C[col + name_a], C[col + name_b])).sum()
            final[col] = (1 - np.isclose(C[col + name_a], C[col + name_b])).sum()
        for col in objs:
            final[col] = (C[col + name_a] != C[col + name_b]).sum()

        # write it out
        unmerged.to_csv(os.path.pardir(log) + 'unmerged.csv', index=False)
        pd.DataFrame(final).to_csv(os.path.pardir(log) + 'merged_no_match.csv', index=False)

        # compare floats & ints for same column
        f.write('Float compare delta...')
        f.write(str(floats))
        f.write('\n')
        for col in floats:
            f.write('Checking column: {}'.format(col))
            delta = C[col + name_b] - C[col + name_a]
            dchg = (delta / C[col + name_a]).round(3)
            check = pd.concat([delta.describe(), dchg.describe()],
                              keys=['Delta', '% Change'], axis=1)
            f.write(str(check))
            f.write('\n')

        f.write('Int compare delta...')
        f.write(str(ints))
        f.write('\n')
        for col in ints:
            f.write('Checking column: {}'.format(col))
            delta = C[col + name_b] - C[col + name_a]
            dchg = (delta / C[col + name_a]).round(3)
            check = pd.concat([delta.describe(), dchg.describe()],
                              keys=['Delta', '% Change'], axis=1)
            f.write(str(check))
            f.write('\n')


## -------------------------------------------------------------------------- ##
## Cleaner Functions
## -------------------------------------------------------------------------- ##

def clean_raw_cols(data, kwargs):
    """
    Convert the columns in a dataframe to not contain special chars
        (special chars can be found if a column is encrypted on payer side)
    @data: dataframe
    returns: df with colnames still in raw format but without special chars
    """
    # Convert columns attribute
    data.columns = [re.sub('[^A-Za-z0-9]+', '', x) for x in data.columns]
    return data


def strip_whitespace_cols(data, kwargs):
    """
    Removes leading / trailing whitespace from columns
    (assuming columns have been read in as strings)
    @data: dataframe
    returns: df with string columns stripped of whitespace
    """
    # Strip cols
    for col in list(data.columns):

        try:
            data.loc[:, col] = data[col].str.strip()
            data.loc[data[col] == "", col] = np.NaN
        except AttributeError as TE:
            continue

    return data


def _lsdir(self, path, refresh=False):
    if path.startswith('s3://'):
        path = path[len('s3://'):]
    path = path.rstrip('/')
    bucket, prefix = split_path(path)
    prefix = prefix + '/' if prefix else ""
    if path not in self.dirs or refresh:
        try:
            pag = self.s3.get_paginator('list_objects_v2')
            it = pag.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/',
                              **self.req_kw)
            files = []
            dirs = []
            for i in it:
                dirs.extend(i.get('CommonPrefixes', []))
                files.extend(i.get('Contents', []))
            if dirs:
                files.extend([{'Key': l['Prefix'][:-1], 'Size': 0,
                               'StorageClass': "DIRECTORY"} for l in dirs])
            files = [f for f in files if len(f['Key']) > len(prefix)]
            for f in files:
                f['Key'] = '/'.join([bucket, f['Key']])
        except ClientError:
            # path not accessible
            files = []
        self.dirs[path] = files
    return self.dirs[path]

S3 = S3FileSystem()


## -------------------------------------------------------------------------- ##
## Email
## -------------------------------------------------------------------------- ##


def password():
    """returns the configuration, and password keys in the .remedy.ini file"""
    return CONFIG.get('configuration', 'password')

# def zip(src, dst):
#     zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
#     abs_src = os.path.abspath(src)
#     for dirname, subdirs, files in os.walk(src):
#         for filename in files:
#             absname = os.path.abspath(os.path.join(dirname, filename))
#             arcname = absname[len(abs_src) + 1:]
#             zf.write(absname, arcname)
#     zf.close()


# def email_writer(password):
#     src = CONFIG.paths['src'] + now()
#     dst = CONFIG.paths['dst'] + now()
#     lpanda = {"lpanda@remedypartners.com": "Message"}
#     for x in [lpanda]:
#         for address, message in x.items():
#             fromaddr = "lpanda@remedypartners.com"
#             toaddr = address
#             msg = MIMEMultipart()
#             msg['From'] = fromaddr
#             msg['To'] = toaddr
#             msg['Subject'] = message
#             body = "This email and it's contents were automatically produced and sent to you for your reference."
#             msg.attach(MIMEText(body, 'plain'))

            # # if you want to attach a file
            # filename = now() + CONFIG.paths['filename']
            # # zip the file before running this
            # attachment = open(CONFIG.paths['attachment'] + now() + ".zip", "rb")
            # ftype, encoding = 'application/zip', None
            # part = MIMEBase('application', 'octet-stream')
            # part.set_payload((attachment).read())
            # encoders.encode_base64(part)
            # part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
            # msg.attach(part)

            # # start an email server
            # server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.starttls()
            # server.login(fromaddr, password)
            # text = msg.as_string()
            # server.sendmail(fromaddr, toaddr, text)
            # server.quit()



## -------------------------------------------------------------------------- ##
## Compare data sources
## -------------------------------------------------------------------------- ##


def convert_data_types_source(source, config):
    """ converting data types of the original file """
    rename = dict(zip(config['orig_name'], config['rename']))
    config = config.loc[config.orig_dtype == 'object']
    dtypes = dict(zip(config['orig_name'], config['orig_dtype']))
    data = pd.read_csv(source, dtype=dtypes)
    data = data.rename(columns=rename)
    dates = [x for x in config['rename'].values if ('date' in x) or ('dod' in x) or ('dob' in x)]
    for i in dates:
        data[i] = pd.to_datetime(data[i])
    return data

def convert_data_types_new(new, config):
    """ convert data types of a new file"""
    config = config.loc[config.db_dtype == 'object']
    dtypes = dict(zip(config['rename'], config['db_dtype']))
    dates = [x for x in config['rename'].values if ('date' in x) or ('dod' in x) or ('dob' in x)]
    data = pd.read_csv(new, dtype=dtypes, parse_dates=dates)
    for i in dates:
        data[i] = pd.to_datetime(data[i])
    return data

def get_diff_cost(df, col_a, col_b):
    """get the differences between column a and column b"""
    df['diff_bw_cols'] = abs(df[col_a].fillna(0) - df[col_b].fillna(0))
    df['Diff Category'] = np.where(df.diff_bw_cols>10000, 'More than $10000',
                              np.where(df.diff_bw_cols>1000, 'Less than $10000',
                                       np.where(df.diff_bw_cols>100, 'Less than $1000',
                                                np.where(df.diff_bw_cols>=1, 'Less than $100', 'Less than $1'))))
    diff_cat = df.groupby('Diff Category')\
                 .agg({col_a: 'count'})\
                 .rename(columns={col_a: 'row_count'})\
                 .reset_index()\
                 .sort_values(by='Diff Category')\
                 .assign(name = col_a)\
                 .loc[:, ['name', 'Diff Category', 'row_count']]

    diff_cat['row_count'] = ['{:,.0f}'.format(x) for x in diff_cat['row_count']]
    return diff_cat


def get_diff_count_measures(df, col_a, col_b):
    """get the differences between column a and column b"""
    df['diff_bw_cols'] = abs(df[col_a].fillna(0) - df[col_b].fillna(0))
    df['Diff Category'] = np.where(df.diff_bw_cols>90, 'More than 90',
                              np.where(df.diff_bw_cols>15, 'Less than 90',
                                       np.where(df.diff_bw_cols>5, 'Less than 15',
                                                np.where(df.diff_bw_cols>=1, 'Less than 5', 'Less than 1'))))
    diff_cat = df.groupby('Diff Category')\
                 .agg({col_a: 'count'})\
                 .rename(columns={col_a: 'row_count'})\
                 .reset_index()\
                 .sort_values(by='Diff Category')\
                 .assign(name = col_a)\
                 .loc[:, ['name', 'Diff Category', 'row_count']]
    return diff_cat


def compare_dfs(A, B, name_a='', name_b='', writer_dir='', vn=now()):
    """ Compare two dataframes, A & B. """
    checks = {}
    wb = pd.ExcelWriter(writer_dir + 'comparison-{}.xlsx'.format(vn))
    title = wb.book.add_format({'bold':True, 'align':'left', 'valign':'vcenter', 'bg_color': '#D3D3D3'})
    print('finished making comparison wb\n')

    # compare columns
    aminusb = set(A.columns).difference(set(B.columns))
    pd.Series(list(aminusb), name='{} columns'.format(name_a)).to_frame()\
        .to_excel(wb, sheet_name='Summary', startcol=0, startrow=12)

    bminusa = set(B.columns).difference(set(A.columns))
    pd.Series(list(bminusa), name='{} columns'.format(name_b)).to_frame()\
        .to_excel(wb, sheet_name='Summary', startcol=4, startrow=12)

    minus = aminusb.union(bminusa)
    print('finished column comparison \n')

    # compare rows after merging
    A = A.loc[A.system_id != 'ZZ704']
    merged = pd.merge(A, B, on=['episode_id', 'bpid'], how='outer', indicator=True, suffixes=[name_a, name_b])
    merge_stats = wb.sheets['Summary']
    merge_stats.merge_range('A1:E1', 'Comparing {} and {} \n\n'.format(name_a, name_b), title)
    merge_stats.merge_range('A2:E2', now(), title)
    merge_stats.merge_range('A3:E3', 'Shapes: {} vs. {}\n\n'.format(A.shape, B.shape))
    merge_stats.merge_range('A5:E5', 'Merge Statistics', title)
    merge_stats.merge_range('A12:D12', 'Columns in {} but not in {}\n'.format(name_a, name_b), title)
    merge_stats.merge_range('E12:H12', 'Columns in {} but not in {}\n'.format(name_b, name_a), title)

    value_counts(merged._merge)\
        .reset_index()\
        .replace('left_only', name_a)\
        .replace('right_only', name_b)\
        .set_index('index')\
        .to_excel(wb, sheet_name='Summary', startcol=1, startrow=5)
    print('finished rows comparison \n')

    # compare data types
    dtypes = pd.concat([A.dtypes.sort_index(), B.dtypes.sort_index()], keys=[name_a, name_b], axis=1, sort=True)
    nulls = pd.concat([A.isnull().sum().sort_index(), B.isnull().sum().sort_index()], keys=[name_a, name_b], axis=1, sort=True)
    totals = pd.concat([dtypes, nulls], keys=['dtypes', 'count nulls'], axis=1, sort=True)
    totals['abs diff nulls'] = np.abs(totals[('count nulls', name_a)] - totals[('count nulls', name_b)])
    diff = totals[('count nulls', name_a)] - totals[('count nulls', name_b)]
    totals['more nulls?'] = np.where((diff == 0) | diff.isnull(), '', np.where(diff > 0, name_a, name_b))
    totals['total rows'] = A.shape[0]
    totals = totals.sort_values([('abs diff nulls', '')], ascending=False)
    totals.to_excel(wb, sheet_name='Datatypes & Nulls Comparison', startcol=1, startrow=3)
    ws = wb.sheets['Datatypes & Nulls Comparison']
    ws.merge_range('A1:I2', 'Datatypes & Nulls Comparison', title)
    print('finished dtypes & nulls comparison \n')

    # select datatypes for columns in both
    floats = dtypes.loc[(dtypes == 'float64').all(axis=1)].index.tolist()
    ints = dtypes.loc[(dtypes == 'int64').all(axis=1)].index.tolist()
    objs = dtypes.loc[(dtypes == 'object').all(axis=1)].index.tolist()

    # subset to rows in both but spit out others:
    unmerged = merged.loc[merged._merge != 'both']
    if not unmerged.empty: unmerged.to_excel(wb, sheet_name='Unmerged Rows')
    duped = B.loc[B.duplicated(['episode_id', 'bpid'], keep=False)]
    if not duped.empty: duped.to_excel(wb, sheet_name='Duplicated Rows')

    C = merged.loc[merged._merge == 'both']
    print('finished subsetting to merged rows \n')

    # compare overall costs
    costs = [x for x in floats if ('price' in x) or ('amount' in x) or ('npra' in x)]
    unmatched_costs = []
    for i,j in enumerate(costs):
        mask = ~np.isclose(C[j+name_a], C[j+name_b], atol=10)
        subs = C.loc[mask, ['episode_id', 'bpid', j+name_a, j+name_b]]\
                .rename(columns={j+name_a: name_a, j+name_b: name_b})\
                .assign(col_name=j)
        if not subs.empty: unmatched_costs.append(subs.sample(10))
        get_diff_cost(C, j+name_a, j+name_b)\
            .to_excel(wb, sheet_name='Cost Columns Comparison', index=False, startrow=3+(i*7), startcol=1)

    if unmatched_costs:
        pd.concat(unmatched_costs, ignore_index=True)\
          .to_excel(wb, sheet_name='Cost Columns Comparison', index=False, startcol=8, startrow=3)

    ws = wb.sheets['Cost Columns Comparison']
    ws.merge_range('A1:E2', 'Examples of Differences in Cost Fields', title)
    ws.merge_range('H1:N2', 'Detail Differences in Cost Fields', title)
    print('finished cost comparison \n')

    # compare string values
    data_diffs = {}
    for i, col in enumerate(objs):
        if (col in ['episode_id', 'bpid']):
            pd.DataFrame({'name': [col], 'result': ['column was a merge key, not counted']}, index=[0])\
              .to_excel(wb, sheet_name='String Columns Comparison', index=False, startrow=4+(i*7), startcol=1)
            continue

        notsame = (C[col + name_a] != C[col + name_b]) & (C[col + name_a].notnull() & C[col + name_b].notnull())
        notsame |= ((C[col + name_a].isnull() & C[col + name_b].notnull()) | (C[col + name_a].notnull() & C[col + name_b].isnull()))
        data_diffs[col] = notsame.sum()

        if notsame.sum() > 1:
            C.loc[C[col + name_a] != C[col + name_b], [col + name_a, col + name_b]]\
             .sample(5)\
             .to_excel(wb, sheet_name='String Columns Comparison', index=False, startrow=4+(i*7), startcol=1)
        else:
            pd.DataFrame({'name': [col], 'result': ['0 rows have a difference']}, index=[0])\
                .to_excel(wb, sheet_name='String Columns Comparison', index=False, startrow=4+(i*7), startcol=1)

    ws = wb.sheets['String Columns Comparison']
    ws.merge_range('A1:E2', 'Examples of Differences in Text Fields', title)
    print('finished string column comparison \n')

    # total counts of differences in text fields
    pd.DataFrame(data_diffs, index=[0])\
      .T\
      .rename(columns={0:'count_different_rows'})\
      .sort_values('count_different_rows', ascending=False)\
      .assign(total_rows = C.shape[0])\
      .to_excel(wb, sheet_name='Datatypes & Nulls Comparison', startcol=11, startrow=4)
    ws = wb.sheets['Datatypes & Nulls Comparison']
    ws.merge_range('K1:N2', 'String Column Value Differences', title)
    print('finished string column diff aggregate summary \n')

    # compare days metric
    counts = [x for x in B if ('day' in x) or ('stays' in x) or ('ilos' in x) or ('count' in x) or ('elos' in x)]
    unmatched_counts = []
    for i,j in enumerate(counts):
        if j not in A:
            pd.DataFrame({'name': [j], 'result': ['column was not found in ' + name_a]}, index=[0])\
              .to_excel(wb, sheet_name='Count & Day Comparison', index=False, startrow=4+(i*6), startcol=1)
            continue

        mask = ~np.isclose(C[j+name_a], C[j+name_b], atol=1) & (C[j+name_a].notnull())
        mask |= ~np.isclose(C[j+name_a], C[j+name_b], atol=1) & (C[j+name_b].notnull())
        subs = C.loc[mask, ['episode_id', 'bpid', j+name_a, j+name_b]]\
                .rename(columns={j+name_a: name_a, j+name_b: name_b})\
                .assign(col_name=j)
        if not subs.empty: unmatched_counts.append(subs.sample(10))
        get_diff_count_measures(C, j+name_a, j+name_b)\
            .to_excel(wb, sheet_name='Count & Day Comparison', index=False, startrow=4+(i*6), startcol=1)

    if unmatched_counts:
        pd.concat(unmatched_counts, ignore_index=True)\
          .to_excel(wb, sheet_name='Count & Day Comparison', index=False, startcol=8, startrow=3)
    ws = wb.sheets['Count & Day Comparison']
    ws.merge_range('A1:E2', 'Examples of Differences in Count & Day Fields', title)
    ws.merge_range('H1:N2', 'Detail Differences in Count Fields', title)
    print('finished counts comparison \n')

    # TODO: compare flags
    flags = [x for x in floats + ints if ('flag' in x) or (x in ['network_tier', 'trigger_claim_line_number'])]


    # TODO: compare decimal metrics
    decs = [x for x in B if 'discharge' in x]
    # C.loc[~np.isclose(C[col + name_a], C[col + name_b]), [col + name_a, col + name_b]].sample(5)

    # TODO: compare dates

    # save
    wb.save()
    wb.close()
