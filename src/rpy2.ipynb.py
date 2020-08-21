# %% markdown
# # rpy2 tests
#   > pls use the _**rpy2**_ kernel
#
# %% markdown
# ## import and setup
#
# %% codecell

import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt;
from matplotlib import dates as mdates;
import seaborn as sns;

import rpy2;
%load_ext rpy2.ipython

import rpy2.robjects as ro;

from rpy2.robjects import pandas2ri;
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter;

from rpy2.robjects.packages import importr;

utils = importr("utils");
grdevices = importr('grDevices');
infenergy = importr('infenergy');


def help_r( python_name ):
    print( str( utils.help( python_name.replace('_','.') ) ) );



# %% markdown
# # read and convert data
#
# %% codecell
data = infenergy.get_inf_meter_data("2014-04-30", "2014-05-01"); # as an r dataframe

with localconverter(ro.default_converter + pandas2ri.converter):
  data_converted = ro.conversion.rpy2py( data );

data_converted = data_converted.reset_index( drop=True );
data_converted.time = data_converted.time.dt.tz_convert('GMT');

data_converted.info()


print( data.head() )

data_converted.head()

# %% markdown
# # Test plots
#
# %% codecell
ax = data_converted.join(data_converted.cumkwh.diff().rename('diff')).dropna().plot( x='time', y='diff', figsize=(8,8) );
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"));
ax.set_xlabel('time');
ax.set_ylabel('diff(cumkwh)');
plt.show();

%R example(get.inf.meter.data)
# %% codecell
