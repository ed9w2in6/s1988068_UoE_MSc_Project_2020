# s1988068_UoE_MSc_Project_2020
 Repository for my MSc project at the University of Edinburgh titled:

 _Data Mining Project - Is there a link between Electricity Consumption and weather in the Informatics Forum_


## Downloading Datasets

**Dataset should be download separately due to licensing and permission issue.**



For the MIDAS dataset, download them from The CEDA Archive at [this link](https://catalogue.ceda.ac.uk/uuid/220a65615218d5c9cc9e4785a3234bd0).
Permission must be obtained from CEDA.

For the Infenergy dataset, access to the School of Informatics Local Area Network is needed.
Then, querying of the data can be referred to the GitHub repository [davidcsterratt/infenergy](https://github.com/davidcsterratt/infenergy).
_**Alternatively**, with the right permission_, the PostgreSQL database that contains the Infenergy data can be dumped.
_**This is the method used in this project and the codes.**_
For this, referred to my edited version of the GitHub repository [ed9w2in6/infenergy](https://github.com/ed9w2in6/infenergy) to query to data.
Instruction and requirements to use the repository are written, fixes are also added to deal with timezone differences, in order to facilitate work in remote environment.


## About the files

The files at the root directory of this repository are described in this section.

### `README.md`

The markdown file `README.md` is this file that you are currently reading.  It describes **notes and instruction to use the codes in this repository**.

### `.git/`, `.DS_Store` and `.gitattributes`

These directories and files are **not related to the project**.  They can be **ignored**.


### `2016_hourly_plots.ipynb` and `MIDAS_2016_daily_hourly_and_other_locations.ipynb`
The Jupyter notebook files `2016_hourly_plots.ipynb` and `MIDAS_2016_daily_hourly_and_other_locations.ipynb` are used for **initial exploration to choose a weather station in Edinburgh to use**.

### `extraction.ipynb` and `Combining.ipynb`
The Jupyter notebook files `extraction.ipynb` and `Combining.ipynb` are used for the **pre-preocessing** of the **MIDAS and Infenergy dataset**.  `extraction.ipynb` is used for **data extraction of the chosen weather station -- Gogar Bank** from the larger complete MIDAS `UK Hourly Weather data` and `UK Hourly Rainfall data`.  `Combining.ipynb` is used to **import Infenergy dataset from a local PostgreSQL server** created from a dump of the one at the LAN of School of Informatics; `Combining.ipynb` is also used to combine the Infenergy dataset with the extracted Gogar Bank weather station data to create a **unified dataset of weather variables and electricity consumption of the Informatics Forum**.

### `environment.yml`
The YAML file `environment.yml` is created using the command `conda env export > environment.yml`.  It can be used to **recreate the Conda environment via Miniconda** using the following command.
```shell
conda env create -f environment.yml
```
Note that the environment are originally setup in a computer with the following **system information**:

> MacOS version: 10.15.6（19G73）
>
> zsh 5.7.1 (x86_64-apple-darwin19.0)
>
> conda 4.8.4

You **MUST** adjust the environment accordingly **if there are any errors**.

### `src/`

The directory `src/` contains the **main code used to produce plots and to fit models, also a testing code to ensure environment is correctly set up**.

#### `src/.DS_Store`

This file is **not related to the project**.  It can be **ignored**.

#### `src/main_EDA_and_Modelling.ipynb` and `src/r_environment_setup_test.ipynb`

The Jupyter notebook file `src/r_environment_setup_test.ipynb` should be ran to **confirm the environment is correctly setup**, there should be no errors and a plot of the hourly electricity consumption of the day 2014-04-30 should be plotted.

The Jupyter notebook file `src/main_EDA_and_Modelling.ipynb` **contains all the codes ever used for producing the figures in the report, and all the models trained** for the projects.



## Other Notes

Note that the file paths for data in all codes should be changed accordingly to match the paths after the datasets are downloaded.
