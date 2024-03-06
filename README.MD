# Pairs Trading via Unsupervised Learning

The notebook with the experiments is available [here](./pairs_unsupervised.ipynb)
The study is a replication and enhancement of the clustering proposed by Han(2022), with our own KMeans Optimization

## Dataset

Data was harvested from Dacheng Xiu's web-site (https://dachxiu.chicagobooth.edu/download/datashare.zip), it is in GBs of size. To observe the data structure and do your own tests, see the provided samples from 2019 to 2021: *"./data/sample_historic_characteristics.csv"*
Quality is questionable, and clean up was required - see notebook. The CRSP and various securities data was unavailable to me at the time of writing, there was no way to map PERMNO to SnP constituants nor make a reversal benchmark mentioned in the paper.

In abscence of adjusted return (accessible from WRDS), the momentum was used as a proxy.

## Firm Characteristics Dataset Description:
1.DATE: The end day of each month (YYYYMMDD)
2.permno: CRSP Permanent Company Number
3-96. 94 Lagged Firm Characteristics (Details are in the appendix of Han(2022)) - lagged as these are released by CRSP with a delay, which we I assume is 1 month.
97.sic2: The first two digits of the Standard Industrial Classification code on DATE

# Credits and Citations

The models were inspired by the paper:

```BibTeX
@article{han2023pairs,
  title={Pairs trading via unsupervised learning},
  author={Han, Chulwoo and He, Zhaodong and Toh, Alenson Jun Wei},
  journal={European Journal of Operational Research},
  volume={307},
  number={2},
  pages={929--947},
  year={2023},
  publisher={Elsevier}
}
```

Cite these papers if using their datasets:

```bibtex
@article{gu2020empirical,
  title={Empirical asset pricing via machine learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies},
  volume={33},
  number={5},
  pages={2223--2273},
  year={2020},
  publisher={Oxford University Press}
}
```

```bibtex
@article{gu2021autoencoder,
  title={Autoencoder asset pricing models},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={Journal of Econometrics},
  volume={222},
  number={1},
  pages={429--450},
  year={2021},
  publisher={Elsevier}
}
```