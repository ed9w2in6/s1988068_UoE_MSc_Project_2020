# s1988068_UoE_MSc_Project_2020
 Repository for my MSc project at the Univeristy of Edinburgh titled:
 
 _Data Mining Project - Is there a link between Electricity Consumption and weather in the Informatics Forum_
 

## Downloading Datasets

**Dataset should be download sepeartely.**



For the MIDAS dataset, download them from The CEDA Archive at [this link](https://catalogue.ceda.ac.uk/uuid/220a65615218d5c9cc9e4785a3234bd0)

For the Infenergy data, access to the School of Informatics Local Area Network is needed.

Then, querying of the data can be referred to this [github repository](https://github.com/davidcsterratt/infenergy)




Alternatively, the PostgreSQL database that contains the Infenergy data can be dumped.

This is the method used in the project and the codes.

For this, referred to this [github repository](https://github.com/ed9w2in6/infenergy) to query to data.

Fixes to deal with timezone differences are implemented as well to facilitate working in remote environment.


## About the files

The files at the root directory of this repository are desribed in this section.

### `README.md`
The markdown file `README.md` at is this file that you are currently reading.  It describes **notes and instruction to use the codes in this repository**.

### `.DS_Store` and `.gitattributes`
These files are are **not related to the project**.  They can be **ignored**.


### `2016_hourly_plots.ipynb` and `MIDAS_2016_daily_hourly_and_other_locations.ipynb`
The Jupyter notebook files `2016_hourly_plots.ipynb` and `MIDAS_2016_daily_hourly_and_other_locations.ipynb` are used for **initial exploration to choose a weather station in Edinburgh to use**.

### `extraction.ipynb` and `Combining.ipynb`
The Jupyter notebook files `extraction.ipynb` and `Combining.ipynb` are used for the **pre-preocessing** of the **MIDAS and Infenergy dataset**.  `extraction.ipynb` is used for **data extraction of the chosen weather station -- Gogar Bank** from the larger complete MIDAS `UK Hourly Weather data` and `UK Hourly Rainfall data`.  `Combining.ipynb` is used to **import Infenergy dataset from a local PostgreSQL server** created from a dump of the one at the LAN of School of Informatics; `Combining.ipynb` is also used to combined the Infenergy dataset with the extracted Gogar Bank weather station data to create a **unified dataset of weather variables and electricity consumption of the Informatics Forum**.

### `environment.yml`
The YAML file `environment.yml` is created using the command `conda env export > environment.yml`.  It can be used to **recreate the Conda environment via Miniconda** using the following command.
```shell
conda env create -f environment.yml
```
Note that the environment are originally setup in the following **system information**:

> MacOS version: 10.15.6（19G73）
> zsh 5.7.1 (x86_64-apple-darwin19.0)
> conda 4.8.4

You **MUST** adjust the environment accordingly **if there are any errors**.

### `src/`

The directory `src/` contontains the **main code used produce plots and to fit models, also a testing code to ensure environment is correctly set up**.

#### `src/main_EDA_and_Modelling.ipynb` and `src/r_environment_setup_test.ipynb`

The Jupyter notebook file `src/r_environment_setup_test.ipynb` should be run to **confirm the environment is correctly setup**, it should be no errors and a plot of the hourly electricity consumption of the day 2014-04-30 should be plotted.
The Jupyter notebook file `src/main_EDA_and_Modelling.ipynb` **contains all the codes ever used for producing the figures in the report, and all the models trained** for the projects.



## Other Notes

Note that the file paths for data in all codes should be changed accordingly to match the paths after the datasets are downloaded.
