This dataset is first studied in Ludwig, J., and Miller, D. L. (2007), _[Does Head Start Improve Children’s Life
Chances? Evidence from a Regression Discontinuity Design](https://harris.uchicago.edu/files/inline-files/QJE_Headstart_2007_0.pdf)"_.

## Explanation of the data

<small>Note: This explanation was taken from Calonico, Sebastian & Cattaneo, Matias & Farrell, Max & Titiunik, Rocío. (2018). _[Regression Discontinuity Designs Using Covariates](https://arxiv.org/pdf/1809.03904v1.pdf)_.</small>

The unit of observation is the U.S. county, the treatment is receiving technical assistance to apply for Head Start funds, and the running variable is the county-level poverty index constructed in 1965 by the federal government based on 1960 census information, with cutoff x = 59.1984. The outcome is the child mortality rate (for children of ages five to nine) due to causes affected by Head Start’s health services component.

There are nine pre-intervention covariates from the 1960 U.S. Census: total population, percentage of black and urban population, and levels and percentages of population in three age groups (children aged 3 to 5, children aged 14 to 17, and adults older than 25).

The data is available at https://sites.google.com/site/rdpackages/replication/.

## Procedures to reproduce

### Programs from Calonico(2018)

Install R.

```bash
sudo apt update
sudo apt install r-base
```

Now type command `R` to enter REPL. Install the required package:

```R
install.packages('rdrobust')
```
Creating a personal library in `~/R/x86_64-pc-linux-gnu-library/3.4`.

```bash
cd data/
Rscript Calonico-Cattaneo-Farrell-Titiunik_2018_RESTAT.R
```

It should have worked, but it turns out that at [line 60](https://github.com/sshockwave/RDD_project/blob/30a26544cdd08a58c0404d180ecc7239a445e26c/headstart/data/Calonico-Cattaneo-Farrell-Titiunik_2018_RESTAT.R#L60) it stops running because `h` is null.